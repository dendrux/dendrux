"""image input at the runtime entry points (run/stream/submit/resume/retry).

The blob catalog, resolver and loop image pipeline belong to other tracks; these
tests use small fakes for them (``ImageStore`` = the real SQLAlchemy store plus an
in-memory catalog that records edges, and ``ImageLoop`` = ``ReActLoop`` that
accepts and records ``image_runtime`` / ``image_view_state``).
"""

from __future__ import annotations

import ast
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine

from dendrux.agent import Agent
from dendrux.chat import ChatMessage, split_user_content
from dendrux.context_blocks import ContextBlock
from dendrux.errors import (
    BlobNotFoundError,
    ImageFetchError,
    ImageLimitError,
    InvalidImageError,
    UnsupportedModalityError,
)
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.media import Image, ImageDropPolicy, ImagePolicy, UrlPolicy
from dendrux.runtime.replay import (
    ImageSource,
    history_seed,
    initial_image_view,
    rebuild_messages,
)
from dendrux.runtime.runner import _carry_image_view, _resume_image_view
from dendrux.runtime.state import RunEventRecord, SQLAlchemyStateStore, TraceRecord
from dendrux.tool import tool
from dendrux.types import (
    IdempotencyConflictError,
    ImagePart,
    ImageRef,
    ImageViewState,
    LLMResponse,
    Message,
    PauseState,
    Role,
    RunStatus,
    TextPart,
    ToolCall,
    ToolResult,
    UsageStats,
    compute_idempotency_fingerprint,
    parts_digest,
)

if TYPE_CHECKING:
    from dendrux.media._catalog import BlobEdge
    from dendrux.media._store import BlobInfo

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "images"
PNG = (FIXTURES / "zebra42.png").read_bytes()
PNG2 = (FIXTURES / "kiwi77.png").read_bytes()
SRC = Path(__file__).resolve().parents[2] / "src" / "dendrux"


# ---------------------------------------------------------------------------
# Fakes for the B (catalog) and D1 (loop) seams
# ---------------------------------------------------------------------------


def _scope(tenant_id: str | None) -> str:
    return "-" if tenant_id is None else "t:" + tenant_id


class ImageStore(SQLAlchemyStateStore):
    """Real store + in-memory ``BlobCatalog``; records every edge write."""

    def __init__(self, engine: Any) -> None:
        super().__init__(engine)
        self.rows: dict[tuple[str, str], tuple[BlobInfo, bytes | None]] = {}
        self.edges: list[tuple[str, BlobEdge]] = []  # (run_id, edge)
        self.edge_calls: list[tuple[str, str]] = []  # (method, run_id)
        self.tenants: dict[str, str | None] = {}

    # --- edge-writing record methods (B's kwargs) ---
    def _check(self, run_id: str, scope: str | None, edges: Any) -> None:
        run_scope = _scope(self.tenants.get(run_id))
        for edge in edges:
            if scope != run_scope or edge.tenant_scope != run_scope:
                raise BlobNotFoundError(None)
            row = self.rows.get((edge.tenant_scope, edge.sha256))
            if row is None or row[0].state != "live":
                raise BlobNotFoundError(None)

    def _add(self, method: str, run_id: str, edges: Any) -> None:
        self.edge_calls.append((method, run_id))
        keys = {(r, e.owner_kind, e.owner_id, e.ordinal) for r, e in self.edges}
        for edge in edges:
            if (run_id, edge.owner_kind, edge.owner_id, edge.ordinal) not in keys:
                self.edges.append((run_id, edge))

    async def create_run(self, run_id: str, agent_name: str, **kwargs: Any) -> Any:
        refs = kwargs.pop("blob_refs", None)
        scope = kwargs.pop("blob_refs_scope", None)
        self.tenants.setdefault(run_id, kwargs.get("tenant_id"))
        if refs:
            self._check(run_id, scope, refs)
        result = await super().create_run(run_id, agent_name, **kwargs)
        if refs and result.outcome == "created":
            self._add("create_run", run_id, refs)
        return result

    async def save_trace(self, run_id: str, *args: Any, **kwargs: Any) -> Any:
        refs = kwargs.pop("blob_refs", None)
        scope = kwargs.pop("blob_refs_scope", None)
        if refs:
            self._check(run_id, scope, refs)
        result = await super().save_trace(run_id, *args, **kwargs)
        if refs:
            self._add("save_trace", run_id, refs)
        return result

    async def save_tool_call(self, run_id: str, *args: Any, **kwargs: Any) -> Any:
        refs = kwargs.pop("blob_refs", None)
        scope = kwargs.pop("blob_refs_scope", None)
        if refs:
            self._check(run_id, scope, refs)
        result = await super().save_tool_call(run_id, *args, **kwargs)
        if refs:
            self._add("save_tool_call", run_id, refs)
        return result

    async def submit_and_claim(self, run_id: str, **kwargs: Any) -> bool:
        refs = kwargs.pop("blob_refs", None)
        scope = kwargs.pop("blob_refs_scope", None)
        if refs:
            record = await self.get_run(run_id)
            self.tenants[run_id] = record.tenant_id if record else None
            self._check(run_id, scope, refs)
        won = await super().submit_and_claim(run_id, **kwargs)
        if won and refs:
            self._add("submit_and_claim", run_id, refs)
        return won

    # --- BlobCatalog ---
    async def get_current(self, scope: str, sha256s: Any) -> dict[str, BlobInfo]:
        out = {}
        for sha in sha256s:
            row = self.rows.get((scope, sha))
            if row is not None and row[0].state in ("live", "purging", "purged"):
                out[sha] = row[0]
        return out

    async def touch_live(self, scope: str, sha256s: Any) -> dict[str, BlobInfo]:
        return {
            sha: info
            for sha, info in (await self.get_current(scope, sha256s)).items()
            if info.state == "live"
        }

    async def get_row(self, scope: str, sha256: str) -> BlobInfo | None:
        row = self.rows.get((scope, sha256))
        return row[0] if row else None

    async def insert_row(self, scope, sha256, info, *, state, generation, data) -> bool:
        if (scope, sha256) in self.rows:
            return False
        self.rows[(scope, sha256)] = (replace(info, state=state, generation=generation), data)
        return True

    async def promote(self, scope, sha256, generation) -> bool:
        info, data = self.rows[(scope, sha256)]
        if info.state != "pending" or info.generation != generation:
            return False
        self.rows[(scope, sha256)] = (replace(info, state="live"), data)
        return True

    async def adopt_pending(self, *a: Any, **k: Any) -> bool:
        return False

    async def revive(self, *a: Any, **k: Any) -> bool:
        return False

    async def mark_deleting(self, *a: Any, **k: Any) -> str | None:
        return None

    async def delete_row(self, *a: Any, **k: Any) -> bool:
        return False

    async def mark_purging(self, scope, sha256, reason) -> str | None:
        info, _ = self.rows[(scope, sha256)]
        self.rows[(scope, sha256)] = (replace(info, state="purged"), None)
        return info.generation

    async def finish_purge(self, *a: Any, **k: Any) -> None:
        return None

    async def stuck_rows(self, *a: Any, **k: Any) -> list[BlobInfo]:
        return []

    async def read_blob_data(self, key: Any) -> bytes | None:
        row = self.rows.get((key.tenant_scope, key.sha256))
        return row[1] if row else None

    async def add_blob_refs(self, run_id: str, run_scope: str, edges: Any) -> None:
        self.tenants.setdefault(run_id, (await self.get_run(run_id)).tenant_id)
        self._check(run_id, run_scope, edges)
        self._add("add_blob_refs", run_id, edges)

    async def run_references(self, run_scope, run_id, sha256) -> bool:
        return any(r == run_id and e.sha256 == sha256 for r, e in self.edges)

    async def blob_references(self, scope, sha256) -> list[Any]:
        return []

    async def delete_orphan_edges(self, limit: int) -> int:
        return 0

    # --- helpers ---
    def edges_of(self, run_id: str, kind: str | None = None) -> list[BlobEdge]:
        return [e for r, e in self.edges if r == run_id and (kind is None or e.owner_kind == kind)]


class FakeResolver:
    def __init__(self, catalog: ImageStore, blob_store: Any = None) -> None:
        self.catalog = catalog
        self.blob_store = blob_store

    async def read(self, scope: str, info: BlobInfo) -> bytes:
        from dendrux.media import BlobKey

        data = self.catalog.rows[(scope, info.sha256)][1]
        if data is None:
            data = await self.blob_store.get(BlobKey(scope, info.sha256, info.generation))
        assert data is not None
        return data


class ImageLoop(ReActLoop):
    """ReActLoop that accepts (and records) the D1 image kwargs."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict[str, Any]] = []

    def _record(self, method: Any, runtime: Any, state: Any, kwargs: dict[str, Any]) -> None:
        # Snapshot: the loop mutates the live state as requests are sent.
        snapshot = ImageViewState.from_dict(state.to_dict()) if state is not None else None
        self.seen.append(
            {"runtime": runtime, "state": snapshot, "history": kwargs.get("initial_history")}
        )
        # Forward to the built-in loop once it takes the kwargs itself (D1 merged).
        params = __import__("inspect").signature(method).parameters
        if "image_runtime" in params:
            kwargs["image_runtime"] = runtime
            kwargs["image_view_state"] = state

    async def run(self, *args: Any, image_runtime=None, image_view_state=None, **kwargs: Any):
        self._record(ReActLoop.run, image_runtime, image_view_state, kwargs)
        return await super().run(*args, **kwargs)

    def run_stream(self, *args: Any, image_runtime=None, image_view_state=None, **kwargs: Any):
        self._record(ReActLoop.run_stream, image_runtime, image_view_state, kwargs)
        return super().run_stream(*args, **kwargs)


class LegacyLoop(ReActLoop):
    """A custom loop written against the pre-0.4 signature (no image kwargs)."""

    async def run(  # type: ignore[override]
        self,
        *,
        agent: Any,
        provider: Any,
        strategy: Any,
        user_input: Any,
        run_id: Any = None,
        recorder: Any = None,
        notifier: Any = None,
        initial_history: Any = None,
        initial_steps: Any = None,
        iteration_offset: int = 0,
        initial_usage: Any = None,
        provider_kwargs: Any = None,
        output_type: Any = None,
        initial_pii_mapping: Any = None,
        state_store: Any = None,
        budget_carry: Any = None,
    ) -> Any:
        return await super().run(
            agent=agent,
            provider=provider,
            strategy=strategy,
            user_input=user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            iteration_offset=iteration_offset,
            initial_usage=initial_usage,
            provider_kwargs=provider_kwargs,
            output_type=output_type,
            initial_pii_mapping=initial_pii_mapping,
            state_store=state_store,
            budget_carry=budget_carry,
        )

    def run_stream(  # type: ignore[override]
        self,
        *,
        agent: Any,
        provider: Any,
        strategy: Any,
        user_input: Any,
        run_id: Any = None,
        recorder: Any = None,
        notifier: Any = None,
        initial_history: Any = None,
        initial_steps: Any = None,
        iteration_offset: int = 0,
        initial_usage: Any = None,
        provider_kwargs: Any = None,
        output_type: Any = None,
        state_store: Any = None,
        interrupt: Any = None,
        budget_carry: Any = None,
    ) -> Any:
        return super().run_stream(
            agent=agent,
            provider=provider,
            strategy=strategy,
            user_input=user_input,
            run_id=run_id,
            recorder=recorder,
            notifier=notifier,
            initial_history=initial_history,
            initial_steps=initial_steps,
            iteration_offset=iteration_offset,
            initial_usage=initial_usage,
            provider_kwargs=provider_kwargs,
            output_type=output_type,
            state_store=state_store,
            interrupt=interrupt,
            budget_carry=budget_carry,
        )


@pytest.fixture(autouse=True)
def _services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dendrux.agent._blob_services_factory", lambda: FakeServices)


class MemoryCatalog(ImageStore):
    """Fake no-database catalog (B's InMemoryBlobCatalog stand-in)."""

    def __init__(self) -> None:
        self.rows = {}
        self.edges = []
        self.edge_calls = []
        self.tenants = {}
        self.released: list[str] = []

    async def add_blob_refs(self, run_id: str, run_scope: str, edges: Any) -> None:
        self.tenants.setdefault(run_id, None if run_scope == "-" else run_scope[2:])
        self._check(run_id, run_scope, edges)
        self._add("add_blob_refs", run_id, edges)

    def release_run(self, run_id: str) -> None:
        self.released.append(run_id)
        self.edges = [(r, e) for r, e in self.edges if r != run_id]


class FakeServices:
    """Stand-in for B's ``BlobServices`` (the agent's single wiring point)."""

    instances: list[FakeServices] = []

    def __init__(self, catalog: Any, blob_store: Any, memory: Any = None) -> None:
        self.catalog = catalog
        self.blob_store = blob_store
        self.resolver = FakeResolver(catalog, blob_store) if catalog is not None else None
        self.accessor = None
        self.memory = memory
        FakeServices.instances.append(self)

    @classmethod
    def for_state_store(cls, store: Any, *, blob_store: Any = None) -> FakeServices:
        return cls(store if isinstance(store, ImageStore) else None, blob_store)

    @classmethod
    def in_memory(cls, *, blob_store: Any = None) -> FakeServices:
        from dendrux.media import InMemoryBlobStore

        memory = MemoryCatalog()
        return cls(memory, blob_store or InMemoryBlobStore(), memory)


@pytest.fixture
async def engine():
    from dendrux.db.models import Base

    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
def store(engine) -> ImageStore:
    return ImageStore(engine)


@tool(target="client")
async def take_screenshot() -> str:
    """Client screenshot."""
    return "never"


@tool()
async def server_add(a: int, b: int) -> int:
    """Server add."""
    return a + b


def _agent(store: Any, responses: list[LLMResponse], **kw: Any) -> Agent:
    policy = kw.pop("images", ImagePolicy(capability_overrides={"mock": "yes"}))
    return Agent(
        name=kw.pop("name", "screen_agent"),
        prompt="You look at screenshots.",
        provider=kw.pop("provider", None) or MockLLM(responses),
        tools=kw.pop("tools", [take_screenshot]),
        state_store=store,
        loop=kw.pop("loop", None) or ImageLoop(),
        images=policy,
        **kw,
    )


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


async def _count_runs(store: ImageStore) -> int:
    return await store.count_runs()


def _llm_messages(agent: Agent, call: int = -1) -> list[Message]:
    msgs = agent.provider.call_history[call]["messages"]  # type: ignore[union-attr]
    return [m for m in msgs if m.role != Role.SYSTEM]


# ---------------------------------------------------------------------------
# UserContent / ChatMessage
# ---------------------------------------------------------------------------


class TestUserContent:
    def test_plain_string(self) -> None:
        assert split_user_content("hi") == ("hi", ())

    def test_sequence_without_images_is_text_only(self) -> None:
        assert split_user_content(["a", "", "b"]) == ("a\n\nb", ())

    def test_sequence_with_image(self) -> None:
        img = Image.from_bytes(PNG)
        text, items = split_user_content(["look", img, ""])
        assert text == "look"
        assert items == ("look", img)

    @pytest.mark.parametrize("bad", [b"bytes", 3, {"a": 1}, [1], Image.from_bytes(PNG)])
    def test_type_errors(self, bad: Any) -> None:
        with pytest.raises(TypeError):
            split_user_content(bad)

    def test_chat_message_user_parts(self) -> None:
        img = Image.from_bytes(PNG)
        msg = ChatMessage.user(["what is this", img])
        assert msg.content == "what is this"
        assert msg.parts == ("what is this", img)
        assert ChatMessage.user(["only", "text"]).parts is None

    def test_chat_message_assistant_images_rejected(self) -> None:
        from dendrux.chat import ChatRole

        with pytest.raises(ValueError, match="assistant"):
            ChatMessage(ChatRole.ASSISTANT, "x", parts=("x", Image.from_bytes(PNG)))

    def test_chat_message_content_must_match(self) -> None:
        from dendrux.chat import ChatRole

        with pytest.raises(ValueError, match="content"):
            ChatMessage(ChatRole.USER, "other", parts=("x", Image.from_bytes(PNG)))

    def test_image_only_history_turn_is_not_empty(self) -> None:
        from dendrux.chat import validate_chat_history

        validate_chat_history(
            [ChatMessage.user([Image.from_bytes(PNG)]), ChatMessage.assistant("ok")]
        )


# ---------------------------------------------------------------------------
# run(): ingest before create_run, input edges, trace parts, runtime threading
# ---------------------------------------------------------------------------


class TestRunWithImages:
    async def test_run_ingests_and_records_parts(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="ZEBRA42")])
        result = await agent.run(["What does this say?", Image.from_bytes(PNG)], tenant_id="acme")
        assert result.status == RunStatus.SUCCESS
        sha = _sha(PNG)
        # blob stored in the tenant scope
        assert ("t:acme", sha) in store.rows
        # input_data carries raw parts; "input" = projection
        record = await store.get_run(result.run_id)
        assert record.input_data["input"] == "What does this say?"
        parts = record.input_data["dendrux.parts"]
        assert parts[0] == {"type": "text", "text": "What does this say?"}
        assert parts[1]["sha256"] == sha and parts[1]["width"] == 400
        # input edge in the create_run transaction; trace edges come only from the
        # recorder's save_trace (B), never from a separate runner write
        assert ("create_run", result.run_id) in store.edge_calls
        edges = [(e.owner_kind, e.owner_id, e.ordinal) for e in store.edges_of(result.run_id)]
        assert edges[0] == ("input", "input", 1)
        assert set(edges[1:]) <= {("trace", "0", 1)}
        assert ("add_blob_refs", result.run_id) not in store.edge_calls
        assert all(e.tenant_scope == "t:acme" for e in store.edges_of(result.run_id))
        # the user trace keeps the projection and the wire parts in meta
        traces = await store.get_traces(result.run_id)
        assert traces[0].content == "What does this say?"
        assert traces[0].meta["dendrux.parts"] == parts
        # provider saw the parts message
        sent = _llm_messages(agent, 0)[-1]
        assert sent.role == Role.USER and sent.images[0].ref.sha256 == sha
        # runtime threaded explicitly with the run's tenant scope
        seen = agent.loop.seen[-1]  # type: ignore[union-attr]
        assert seen["runtime"].tenant_scope == "t:acme"
        assert seen["runtime"].run_id == result.run_id
        assert seen["state"] == ImageViewState()

    async def test_text_only_run_unchanged(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="hi")])
        result = await agent.run("hello")
        record = await store.get_run(result.run_id)
        assert record.input_data == {"input": "hello"}
        assert store.edges == [] and store.rows == {}
        seen = agent.loop.seen[-1]  # type: ignore[union-attr]
        assert seen["state"] is None and seen["runtime"] is not None
        assert seen["history"] is None

    async def test_context_folding_with_image(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="ok")])
        ctx = [ContextBlock("rules", placement="stable"), ContextBlock("now", placement="dynamic")]
        result = await agent.run([Image.from_bytes(PNG), "caption"], context=ctx)
        sent = _llm_messages(agent, 0)[-1]
        assert isinstance(sent.parts[0], TextPart)
        assert sent.parts[0].text.startswith("STABLE CONTEXT:\nrules")
        assert isinstance(sent.parts[1], ImagePart)
        assert sent.parts[2] == TextPart("caption")
        traces = await store.get_traces(result.run_id)
        assert traces[0].content == "caption"
        assert traces[0].meta["dendrux.parts"][0]["text"].startswith("STABLE CONTEXT:")
        record = await store.get_run(result.run_id)
        assert record.input_data["dendrux.parts"][0]["type"] == "image"

    async def test_non_vision_model_raises_before_create_run(self, store: ImageStore) -> None:
        agent = _agent(
            store, [LLMResponse(text="x")], images=ImagePolicy(capability_overrides={"mock": "no"})
        )
        with pytest.raises(UnsupportedModalityError) as info:
            await agent.run(["hi", Image.from_bytes(PNG)])
        assert info.value.source == "capability" and info.value.image_count == 1
        assert await _count_runs(store) == 0
        assert store.rows == {}

    async def test_custom_provider_without_multimodal_is_no(self, store: ImageStore) -> None:
        try:  # C: MockLLM(image_support=...) defaults to "yes"
            provider = MockLLM([LLMResponse(text="x")], image_support="no")  # type: ignore[call-arg]
        except TypeError:
            provider = MockLLM([LLMResponse(text="x")])
        agent = _agent(store, [], provider=provider, images=ImagePolicy())
        with pytest.raises(UnsupportedModalityError):
            await agent.run([Image.from_bytes(PNG)])
        assert await _count_runs(store) == 0

    async def test_placeholder_policy_does_not_raise_early(self, store: ImageStore) -> None:
        agent = _agent(
            store,
            [LLMResponse(text="x")],
            images=ImagePolicy(on_unsupported="placeholder", capability_overrides={"mock": "no"}),
        )
        result = await agent.run([Image.from_bytes(PNG)])
        assert result.status == RunStatus.SUCCESS

    async def test_oversize_raises_before_create_run(self, store: ImageStore) -> None:
        agent = _agent(
            store,
            [LLMResponse(text="x")],
            images=ImagePolicy(max_image_bytes=100, capability_overrides={"mock": "yes"}),
        )
        with pytest.raises(ImageLimitError):
            await agent.run([Image.from_bytes(PNG)])
        assert await _count_runs(store) == 0 and store.rows == {}

    async def test_unknown_ref_raises_blob_not_found(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="x")])
        ref = ImageRef(sha256="a" * 64, media_type="image/png", byte_length=3, width=1, height=1)
        with pytest.raises(BlobNotFoundError):
            await agent.run(["x", ref])
        assert await _count_runs(store) == 0

    async def test_ref_reuse_is_canonicalised(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="a"), LLMResponse(text="b")])
        first = await agent.run([Image.from_bytes(PNG)])
        record = await store.get_run(first.run_id)
        ref = ImageRef.from_dict(record.input_data["dendrux.parts"][0])
        second = await agent.run(["again", ImagePart(ref, detail="low")])
        parts = (await store.get_run(second.run_id)).input_data["dendrux.parts"]
        assert parts[1]["detail"] == "low"
        wrong = replace(ref, width=999)
        with pytest.raises(InvalidImageError, match="descriptor"):
            await agent.run([wrong])

    async def test_loop_without_image_runtime_raises_type_error(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="x")], loop=LegacyLoop())
        with pytest.raises(TypeError, match="image_runtime"):
            await agent.run([Image.from_bytes(PNG)])
        with pytest.raises(TypeError, match="image_runtime"):
            agent.stream([Image.from_bytes(PNG)])
        assert await _count_runs(store) == 0
        assert store.rows == {}

    async def test_no_database_uses_in_memory_backend(self) -> None:
        agent = Agent(
            prompt="p",
            provider=MockLLM([LLMResponse(text="ok")]),
            loop=ImageLoop(),
            images=ImagePolicy(capability_overrides={"mock": "yes"}),
        )
        history = [ChatMessage.user(["old", Image.from_bytes(PNG2)]), ChatMessage.assistant("ok")]
        result = await agent.run([Image.from_bytes(PNG)], history=history)
        assert result.status == RunStatus.SUCCESS
        services = agent._blob_services(None)
        catalog = services.memory
        infos = [info for info, _ in catalog.rows.values()]
        # external (in-memory) storage: bytes live in the InMemoryBlobStore
        assert {i.storage for i in infos} == {"external"}
        assert services.blob_store.total_bytes == len(PNG) + len(PNG2)
        # input + seed edges registered for the run, then released when it ended
        assert [m for m, r in catalog.edge_calls if r == result.run_id] == ["add_blob_refs"] * 2
        assert catalog.released == [result.run_id]
        assert catalog.edges == []
        assert agent._blob_services(None) is services

    async def test_blobs_accessor_delegates_to_services(self, store: ImageStore) -> None:
        agent = _agent(store, [])

        class _Accessor:
            async def read(self, image: Any, *, tenant_id: Any = None) -> bytes:
                return b"bytes:" + str(tenant_id).encode()

        agent._blob_services(store).accessor = _Accessor()
        assert await agent.blobs.read("a" * 64, tenant_id="acme") == b"bytes:acme"

    async def test_store_without_catalog_raises_persistence_unsupported(self, engine) -> None:
        from dendrux.errors import ImagePersistenceUnsupportedError

        agent = _agent(SQLAlchemyStateStore(engine), [LLMResponse(text="x")])
        if agent._blob_services(agent._state_store).catalog is not None:
            pytest.skip("built-in store implements the catalog (B merged)")
        with pytest.raises(ImagePersistenceUnsupportedError):
            await agent.run([Image.from_bytes(PNG)])


class TestIdempotencyWithImages:
    async def test_same_image_hits_cache_other_image_conflicts(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="a"), LLMResponse(text="b")])
        first = await agent.run(["q", Image.from_bytes(PNG)], idempotency_key="k1")
        again = await agent.run(["q", Image.from_bytes(PNG)], idempotency_key="k1")
        assert again.run_id == first.run_id
        assert agent.provider.calls_made == 1  # type: ignore[union-attr]
        with pytest.raises(IdempotencyConflictError):
            await agent.run(["q", Image.from_bytes(PNG2)], idempotency_key="k1")

    async def test_fingerprint_adds_parts_digest_only_for_images(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="a")])
        result = await agent.run(["q", Image.from_bytes(PNG)], idempotency_key="k2")
        record = await store.get_run(result.run_id)
        parts = (
            TextPart("q"),
            ImagePart(ImageRef.from_dict(record.input_data["dendrux.parts"][1])),
        )
        fp = compute_idempotency_fingerprint(
            "screen_agent", "q", user_input_parts=parts, skills=None
        )
        text_fp = compute_idempotency_fingerprint("screen_agent", "q", skills=None)
        assert fp != text_fp
        async with store._engine.connect() as conn:  # type: ignore[attr-defined]
            stored = (
                await conn.execute(
                    sql_text("SELECT idempotency_fingerprint FROM agent_runs WHERE id = :i"),
                    {"i": result.run_id},
                )
            ).scalar_one()
        assert stored == fp


# ---------------------------------------------------------------------------
# stream(): async ingest before create_run; errors raised from first __anext__
# ---------------------------------------------------------------------------


class TestStreamWithImages:
    def test_bad_item_type_raises_synchronously(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="x")])
        with pytest.raises(TypeError):
            agent.stream(["x", 42])  # type: ignore[list-item]

    @pytest.mark.parametrize(
        ("policy", "image", "error"),
        [
            (
                ImagePolicy(urls=UrlPolicy(fetch=False), capability_overrides={"mock": "yes"}),
                Image.from_url("https://example.com/a.png"),
                ImageFetchError,
            ),
            (
                ImagePolicy(max_image_bytes=100, capability_overrides={"mock": "yes"}),
                Image.from_bytes(PNG),
                ImageLimitError,
            ),
            (
                ImagePolicy(capability_overrides={"mock": "no"}),
                Image.from_bytes(PNG),
                UnsupportedModalityError,
            ),
        ],
    )
    async def test_errors_raise_on_first_anext(
        self, store: ImageStore, policy: ImagePolicy, image: Image, error: type
    ) -> None:
        agent = _agent(store, [LLMResponse(text="x")], images=policy)
        stream = agent.stream(["look", image])
        with pytest.raises(error):
            async for _ in stream:
                pass
        assert await _count_runs(store) == 0
        assert store.rows == {}

    async def test_stream_success_records_parts(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="ZEBRA42")])
        events = [e async for e in agent.stream(["look", Image.from_bytes(PNG)], tenant_id="t1")]
        assert events[-1].type.value == "run_completed"
        run_id = events[-1].run_result.run_id
        record = await store.get_run(run_id)
        assert record.input_data["dendrux.parts"][1]["sha256"] == _sha(PNG)
        assert [e.owner_kind for e in store.edges_of(run_id)][0] == "input"
        assert ("add_blob_refs", run_id) not in store.edge_calls
        assert agent.loop.seen[-1]["runtime"].tenant_scope == "t:t1"  # type: ignore[union-attr]

    async def test_interleaved_streams_keep_tenant_scopes(self, store: ImageStore) -> None:
        a = _agent(store, [LLMResponse(text="a")], name="a")
        b = _agent(store, [LLMResponse(text="b")], name="b")
        sa = a.stream([Image.from_bytes(PNG)], tenant_id="ta").__aiter__()
        sb = b.stream([Image.from_bytes(PNG)], tenant_id="tb").__aiter__()
        done_a = done_b = False
        ids: dict[str, str] = {}
        while not (done_a and done_b):
            for name, it in (("a", sa), ("b", sb)):
                if (name == "a" and done_a) or (name == "b" and done_b):
                    continue
                try:
                    event = await it.__anext__()
                    if event.run_result is not None:
                        ids[name] = event.run_result.run_id
                except StopAsyncIteration:
                    if name == "a":
                        done_a = True
                    else:
                        done_b = True
        assert {s for s, _ in store.rows} == {"t:ta", "t:tb"}
        assert {e.tenant_scope for e in store.edges_of(ids["a"])} == {"t:ta"}
        assert {e.tenant_scope for e in store.edges_of(ids["b"])} == {"t:tb"}

    async def test_concurrent_runs_across_tenants(self, store: ImageStore) -> None:
        agents = [_agent(store, [LLMResponse(text=str(i))], name=f"a{i}") for i in range(8)]
        results = await asyncio.gather(
            *(
                ag.run([Image.from_bytes(PNG if i % 2 else PNG2)], tenant_id=f"t{i % 2}")
                for i, ag in enumerate(agents)
            )
        )
        for i, result in enumerate(results):
            scope = f"t:t{i % 2}"
            assert {e.tenant_scope for e in store.edges_of(result.run_id)} == {scope}
            assert all((scope, e.sha256) in store.rows for e in store.edges_of(result.run_id))


# ---------------------------------------------------------------------------
# history= images (seeded: already sent, seed edges, no capability check)
# ---------------------------------------------------------------------------


class TestHistoryImages:
    async def test_history_images_are_seeded(self, store: ImageStore) -> None:
        agent = _agent(
            store,
            [LLMResponse(text="ok")],
            images=ImagePolicy(capability_overrides={"mock": "no"}),
        )
        history = [
            ChatMessage.user(["earlier", Image.from_bytes(PNG)]),
            ChatMessage.assistant("seen"),
        ]
        # history images never fail the capability check (they count as sent)
        result = await agent.run("and now?", history=history, tenant_id="acme")
        assert result.status == RunStatus.SUCCESS
        seed = store.edges_of(result.run_id, "seed")
        assert [(e.owner_id, e.ordinal, e.sha256) for e in seed] == [("seed", 0, _sha(PNG))]
        state = agent.loop.seen[-1]["state"]  # type: ignore[union-attr]
        assert state.sent_through == 1
        sent = _llm_messages(agent, 0)
        # D1 renders the history image as the unsupported placeholder on this model
        assert sent[0].images or "not shown" in sent[0].content
        assert sent[-1].content == "and now?"

    async def test_history_images_enter_prefix_seed_and_fingerprint(self) -> None:
        ref = ImageRef(sha256="b" * 64, media_type="image/png", byte_length=3, width=1, height=1)
        text_history = [Message(role=Role.USER, content="x")]
        parts_history = [Message.with_parts(Role.USER, [TextPart("x"), ImagePart(ref)])]
        assert history_seed(text_history) != history_seed(parts_history)
        assert history_seed(text_history) == history_seed([Message(role=Role.USER, content="x")])
        fp_text = compute_idempotency_fingerprint("a", "q", history=text_history)
        fp_parts = compute_idempotency_fingerprint("a", "q", history=parts_history)
        assert fp_text != fp_parts


# ---------------------------------------------------------------------------
# Client tools: submit_tool_results / resume(tool_results=) with images (§6)
# ---------------------------------------------------------------------------


def _screenshot_responses(final: str = "The code word is ZEBRA42.") -> list[LLMResponse]:
    return [
        LLMResponse(text=None, tool_calls=[ToolCall(name="take_screenshot", params={})]),
        LLMResponse(text=final),
    ]


class TestClientToolImages:
    async def test_canonical_sample(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run(
            "Take a screenshot and tell me the code word shown in it.", tenant_id="acme"
        )
        assert paused.status == RunStatus.WAITING_CLIENT_TOOL
        run_id = paused.run_id
        raw = await store.get_pause_state(run_id)
        assert "image_view" not in raw  # v0.3.0 shape at step 2
        call_id = raw["pending_tool_calls"][0]["id"]
        iteration = raw["iteration"]

        # Resume in a "new process": a fresh Agent + provider on the same store.
        agent2 = _agent(store, [_screenshot_responses()[1]])
        result = await agent2.submit_tool_results(
            run_id,
            [
                ToolResult(
                    name="take_screenshot",
                    call_id=call_id,
                    payload=json.dumps("Captured."),
                    images=(Image.from_bytes(PNG),),
                )
            ],
        )
        assert result.status == RunStatus.SUCCESS
        sha = _sha(PNG)
        sub = store.edges_of(run_id, "submission")
        assert [(e.owner_id, e.ordinal, e.tenant_scope) for e in sub] == [
            (f"{iteration}:{call_id}", 0, "t:acme")
        ]
        assert ("submit_and_claim", run_id) in store.edge_calls
        # provider saw the TOOL message with the image part
        tool_msg = next(m for m in _llm_messages(agent2) if m.role == Role.TOOL)
        assert tool_msg.content == '"Captured."'
        assert tool_msg.images[0].ref.sha256 == sha
        events = await store.get_run_events(run_id)
        resumed = next(e for e in events if e.event_type == "run.resumed")
        assert resumed.data["submitted_results"] == [
            {"call_id": call_id, "name": "take_screenshot", "success": True, "image_count": 1}
        ]
        # loop resumed with a fresh view: the old history had no image slots
        state = agent2.loop.seen[-1]["state"]  # type: ignore[union-attr]
        assert state == ImageViewState(sent_through=0)

    async def test_submit_image_error_leaves_run_waiting(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run("shot", tenant_id="acme")
        call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
        agent2 = _agent(store, [], images=ImagePolicy(capability_overrides={"mock": "no"}))
        with pytest.raises(UnsupportedModalityError):
            await agent2.submit_tool_results(
                paused.run_id,
                [
                    ToolResult(
                        name="take_screenshot",
                        call_id=call_id,
                        payload='"x"',
                        images=(Image.from_bytes(PNG),),
                    )
                ],
            )
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_CLIENT_TOOL.value
        assert store.rows == {}
        assert "submitted_tool_results" not in await store.get_pause_state(paused.run_id)

    async def test_direct_resume_with_tool_images(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run("shot")
        call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
        agent2 = _agent(store, [_screenshot_responses()[1]])
        result = await agent2.resume(
            paused.run_id,
            tool_results=[
                ToolResult(
                    name="take_screenshot",
                    call_id=call_id,
                    payload='"x"',
                    images=(Image.from_bytes(PNG),),
                )
            ],
        )
        assert result.status == RunStatus.SUCCESS
        tool_msg = next(m for m in _llm_messages(agent2) if m.role == Role.TOOL)
        assert tool_msg.images

    async def test_resume_stream_with_tool_images_error_raises(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run("shot")
        call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
        agent2 = _agent(
            store,
            [],
            images=ImagePolicy(max_image_bytes=10, capability_overrides={"mock": "yes"}),
        )
        stream = agent2.resume_stream(
            paused.run_id,
            tool_results=[
                ToolResult(
                    name="take_screenshot",
                    call_id=call_id,
                    payload='"x"',
                    images=(Image.from_bytes(PNG),),
                )
            ],
        )
        with pytest.raises(ImageLimitError):
            async for _ in stream:
                pass
        record = await store.get_run(paused.run_id)
        assert record.status == RunStatus.WAITING_CLIENT_TOOL.value

    async def test_resume_stream_with_tool_images(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run("shot")
        call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
        agent2 = _agent(store, [_screenshot_responses()[1]])
        events = [
            e
            async for e in agent2.resume_stream(
                paused.run_id,
                tool_results=[
                    ToolResult(
                        name="take_screenshot",
                        call_id=call_id,
                        payload='"x"',
                        images=(Image.from_bytes(PNG),),
                    )
                ],
            )
        ]
        assert events[-1].type.value == "run_completed"
        assert agent2.loop.seen[-1]["runtime"] is not None  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Clarification input: submit_input / resume(user_input=) with images
# ---------------------------------------------------------------------------


def _clarify_responses() -> list[LLMResponse]:
    from dendrux.types import Clarification  # noqa: F401  (documents the pause kind)

    return [
        LLMResponse(text=None, tool_calls=[ToolCall(name="ask_user", params={"question": "?"})]),
        LLMResponse(text="thanks"),
    ]


async def _human_pause(store: ImageStore, agent: Agent) -> str:
    """Persist a WAITING_HUMAN_INPUT pause directly (loop-independent)."""
    run_id = "01JHUMANPAUSE000000000000"
    await store.create_run(run_id, agent.name, input_data={"input": "q"}, tenant_id="acme")
    pause = PauseState(
        agent_name=agent.name,
        pending_tool_calls=[],
        history=[Message(role=Role.USER, content="q"), Message(role=Role.ASSISTANT, content="?")],
        steps=[],
        iteration=2,
        trace_order_offset=2,
        usage=UsageStats(),
    )
    await store.pause_run(
        run_id,
        status=RunStatus.WAITING_HUMAN_INPUT.value,
        pause_data=pause.to_dict(),
        iteration_count=2,
    )
    return run_id


class TestUserInputImages:
    async def test_submit_input_with_image(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="thanks")])
        run_id = await _human_pause(store, agent)
        result = await agent.submit_input(run_id, ["here it is", Image.from_bytes(PNG)])
        assert result.status == RunStatus.SUCCESS
        sub = store.edges_of(run_id, "submission")
        assert [(e.owner_id, e.ordinal) for e in sub] == [("2:input", 1)]
        last = _llm_messages(agent)[-1]
        assert last.role == Role.USER and last.images and last.content == "here it is"
        events = await store.get_run_events(run_id)
        resumed = next(e for e in events if e.event_type == "run.resumed")
        assert resumed.data["image_count"] == 1
        assert resumed.data["user_input"] == "here it is"

    async def test_submit_input_text_sequence_stays_text(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="thanks")])
        run_id = await _human_pause(store, agent)
        await agent.submit_input(run_id, ["a", "b"])
        last = _llm_messages(agent)[-1]
        assert last.parts is None and last.content == "a\n\nb"

    async def test_resume_with_user_parts(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="thanks")])
        run_id = await _human_pause(store, agent)
        result = await agent.resume(run_id, user_input=[Image.from_bytes(PNG), "see"])
        assert result.status == RunStatus.SUCCESS
        last = _llm_messages(agent)[-1]
        assert isinstance(last.parts[0], ImagePart)


# ---------------------------------------------------------------------------
# Pause carry (N-2)
# ---------------------------------------------------------------------------


def _pause(history: list[Message], view: ImageViewState | None = None) -> PauseState:
    return PauseState(
        agent_name="a",
        pending_tool_calls=[],
        history=history,
        steps=[],
        iteration=1,
        trace_order_offset=0,
        usage=UsageStats(),
        image_view=view,
    )


REF = ImageRef(sha256="c" * 64, media_type="image/png", byte_length=3, width=1, height=1)
IMG_MSG = Message.with_parts(Role.USER, [TextPart("x"), ImagePart(REF)])


class TestCarryImageView:
    def test_carries_a_copy(self) -> None:
        view = ImageViewState(epoch=2, dropped_through=3, sent_through=4, placeholders={0: "drop"})
        new = _carry_image_view(_pause([IMG_MSG]), _pause([IMG_MSG], view))
        assert new.image_view == view and new.image_view is not view
        assert new.to_dict()["image_view"] == view.to_dict()

    def test_text_only_stays_absent(self) -> None:
        new = _carry_image_view(_pause([Message(role=Role.USER, content="x")]), _pause([]))
        assert new.image_view is None and "image_view" not in new.to_dict()

    def test_new_images_after_text_pause_start_a_view(self) -> None:
        old = _pause([Message(role=Role.USER, content="x")])
        new = _carry_image_view(_pause([IMG_MSG]), old)
        assert new.image_view == ImageViewState(sent_through=0)

    def test_resume_view(self) -> None:
        view = ImageViewState(sent_through=1)
        assert _resume_image_view(_pause([IMG_MSG], view), [IMG_MSG]) is view
        assert _resume_image_view(_pause([]), []) is None
        assert _resume_image_view(_pause([IMG_MSG]), [IMG_MSG, IMG_MSG]) == ImageViewState(
            sent_through=1
        )


def _pause_state_calls(directory: Path) -> list[tuple[str, int, set[str]]]:
    out = []
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "PauseState"
            ):
                out.append((path.name, node.lineno, {k.arg for k in node.keywords if k.arg}))
    return out


class TestPauseStateSites:
    def test_runtime_sites_pass_image_view(self) -> None:
        calls = _pause_state_calls(SRC / "runtime")
        assert calls, "expected PauseState( sites in runtime/"
        missing = [(f, line) for f, line, kws in calls if "image_view" not in kws]
        assert missing == []

    def test_loop_sites_pass_image_view(self) -> None:
        calls = _pause_state_calls(SRC / "loops")
        missing = [(f, line) for f, line, kws in calls if "image_view" not in kws]
        assert missing == []


class TestApprovalCarry:
    async def test_approval_then_client_pause_then_resume_carries_view(
        self, store: ImageStore
    ) -> None:
        """approval pause (with a view) -> approve -> client pause -> submit in a new Agent."""
        from dendrux.types import ToolTarget  # noqa: F401

        responses = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(name="server_add", params={"a": 1, "b": 2}),
                    ToolCall(name="take_screenshot", params={}),
                ],
            ),
        ]
        agent = _agent(
            store,
            responses,
            tools=[server_add, take_screenshot],
            require_approval=["server_add"],
        )
        paused = await agent.run(["look", Image.from_bytes(PNG)], tenant_id="acme")
        assert paused.status == RunStatus.WAITING_APPROVAL
        run_id = paused.run_id
        # Simulate D1's pause builder: the approval pause carries the live view.
        view = ImageViewState(epoch=1, dropped_through=0, sent_through=1, placeholders={})
        raw = await store.get_pause_state(run_id)
        raw["image_view"] = view.to_dict()
        async with store._engine.begin() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                sql_text("UPDATE agent_runs SET pause_data = :d WHERE id = :i"),
                {"d": json.dumps(raw), "i": run_id},
            )

        approved = await agent.submit_approval(
            run_id,
            approved=True,
            expected_pause_id=(await store.get_pause_state(run_id))["pause_id"],
        )
        assert approved.status == RunStatus.WAITING_CLIENT_TOOL
        client_raw = await store.get_pause_state(run_id)
        assert client_raw["image_view"] == view.to_dict()
        call_id = client_raw["pending_tool_calls"][0]["id"]

        fresh = _agent(store, [LLMResponse(text="done")], tools=[server_add, take_screenshot])
        result = await fresh.submit_tool_results(
            run_id,
            [
                ToolResult(
                    name="take_screenshot",
                    call_id=call_id,
                    payload='"ok"',
                    images=(Image.from_bytes(PNG2),),
                )
            ],
        )
        assert result.status == RunStatus.SUCCESS
        state = fresh.loop.seen[-1]["state"]  # type: ignore[union-attr]
        assert state.to_dict() == view.to_dict()


# ---------------------------------------------------------------------------
# v0.3.0 pauses resume unchanged (I-02)
# ---------------------------------------------------------------------------


class TestLegacyPause:
    async def test_text_pause_resumes_without_image_state(self, store: ImageStore) -> None:
        agent = _agent(store, _screenshot_responses())
        paused = await agent.run("shot")
        raw = await store.get_pause_state(paused.run_id)
        assert "image_view" not in raw
        call_id = raw["pending_tool_calls"][0]["id"]
        agent2 = _agent(store, [_screenshot_responses()[1]])
        result = await agent2.submit_tool_results(
            paused.run_id, [ToolResult(name="take_screenshot", call_id=call_id, payload='"x"')]
        )
        assert result.status == RunStatus.SUCCESS
        assert agent2.loop.seen[-1]["state"] is None  # type: ignore[union-attr]
        events = await store.get_run_events(paused.run_id)
        resumed = next(e for e in events if e.event_type == "run.resumed")
        assert "image_count" not in resumed.data["submitted_results"][0]
        assert store.edges == []


# ---------------------------------------------------------------------------
# Replay / retry state initialisation (§5.14)
# ---------------------------------------------------------------------------


def _event(kind: str, data: dict[str, Any], seq: int = 0) -> RunEventRecord:
    return RunEventRecord(
        id=f"e{seq}", event_type=kind, sequence_index=seq, iteration_index=0, data=data
    )


class TestInitialImageView:
    def test_fold_precedence_and_scope(self) -> None:
        s1 = ImageSource(
            "t:a",
            [
                _event("image.rejected", {"sha256": "r1", "tenant_scope": "t:a"}, 1),
                _event("image.rejected", {"sha256": "x", "tenant_scope": "t:b"}, 2),
                _event(
                    "image.state",
                    {
                        "described": [{"sha256": "d", "text": "first"}],
                        "guardrail_verdicts": [
                            {
                                "sha256": "g",
                                "guardrail_id": "G",
                                "guardrail_version": "1",
                                "origin": "tool",
                                "tool_name": "shot",
                                "action": "pass",
                                "entities": [],
                            }
                        ],
                        "placeholders": [{"slot": 0, "reason": "drop"}],
                    },
                    3,
                ),
            ],
        )
        s2 = ImageSource(
            "t:a",
            [
                _event("image.rejected", {"sha256": "r1", "tenant_scope": "t:a"}, 1),
                _event("image.rejected", {"sha256": "r2", "tenant_scope": "t:a"}, 2),
                _event("image.state", {"described": [{"sha256": "d", "text": "second"}]}, 3),
                _event("image.view_changed", {"epoch": 2, "dropped_through": 5, "visible": 3}, 4),
            ],
        )
        state = initial_image_view(
            sent_through=7, sources=[s1, s2], tenant_scope="t:a", single_full=False
        )
        assert state.sent_through == 7
        assert state.rejected == [("t:a", "r1"), ("t:a", "r2")]
        assert state.described == {("t:a", "d"): "first"}  # earlier source wins
        assert ("t:a", "g", "G", "1", "tool", "shot") in state.guardrail_verdicts
        # several sources -> fresh drop view, no placeholders
        assert (state.epoch, state.dropped_through, state.placeholders) == (0, 0, {})

    def test_single_full_source_carries_drop_view(self) -> None:
        src = ImageSource(
            "-",
            [
                _event("image.view_changed", {"epoch": 1, "dropped_through": 2}, 1),
                _event("image.state", {"placeholders": [{"slot": 0, "reason": "drop"}]}, 2),
                _event("image.view_changed", {"epoch": 2, "dropped_through": 4}, 3),
                _event(
                    "image.state",
                    {
                        "placeholders": [
                            {"slot": 0, "reason": "rejected"},
                            {"slot": 1, "reason": "purged"},
                        ]
                    },
                    4,
                ),
            ],
        )
        state = initial_image_view(
            sent_through=5, sources=[src], tenant_scope="-", single_full=True
        )
        assert (state.epoch, state.dropped_through) == (2, 4)
        assert state.placeholders == {0: "drop", 1: "purged"}

    def test_other_tenant_source_ignored(self) -> None:
        src = ImageSource("t:b", [_event("image.rejected", {"sha256": "r"}, 1)])
        state = initial_image_view(
            sent_through=0, sources=[src], tenant_scope="t:a", single_full=True
        )
        assert state.rejected == []


class TestRebuildParts:
    def test_rebuild_uses_trace_parts(self) -> None:
        img = REF.to_dict()
        traces = [
            TraceRecord(
                id="1",
                role="user",
                content="raw",
                order_index=0,
                meta={"dendrux.parts": [{"type": "text", "text": "framed"}, img]},
            ),
            TraceRecord(
                id="2",
                role="tool",
                content='"Captured."',
                order_index=1,
                meta={
                    "tool_name": "shot",
                    "call_id": "c1",
                    "dendrux.parts": [{"type": "text", "text": '"Captured."'}, img],
                },
            ),
        ]
        msgs = rebuild_messages(traces, reasoning_for=None)
        assert msgs[0].parts == (TextPart("framed"), ImagePart(REF))
        assert msgs[0].content == "framed"
        assert msgs[1].role == Role.TOOL and msgs[1].images[0].ref == REF
        assert msgs[1].call_id == "c1"
        assert parts_digest(msgs[1].parts)[1][2] == "c" * 64


class TestRetryAndReplay:
    async def test_retry_carries_parts_seeds_and_rejections(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="first")])
        source = await agent.run(["what", Image.from_bytes(PNG)], tenant_id="acme")
        sha = _sha(PNG)
        await store.save_run_event(
            source.run_id,
            event_type="image.rejected",
            iteration_index=1,
            data={"image_id": "img_x", "sha256": sha, "tenant_scope": "t:acme"},
        )
        retry_agent = _agent(store, [LLMResponse(text="second")])
        result = await retry_agent.retry(source.run_id, tenant_id="acme")
        assert result.status == RunStatus.SUCCESS
        record = await store.get_run(result.run_id)
        assert record.input_data["dendrux.parts"][1]["sha256"] == sha
        assert [e.owner_kind for e in store.edges_of(result.run_id)] == ["input", "seed"]
        state = retry_agent.loop.seen[-1]["state"]  # type: ignore[union-attr]
        assert state.sent_through == 1
        assert state.rejected == [("t:acme", sha)]

    async def test_replay_of_image_run_seeds_history(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="one")])
        source = await agent.run(["see", Image.from_bytes(PNG)])
        follow = _agent(
            store,
            [LLMResponse(text="two")],
            images=ImagePolicy(capability_overrides={"mock": "no"}),
        )
        # history images never error, even on a model that cannot take images
        result = await follow.run("and?", runs=[source.run_id], replay_last=1)
        assert result.status == RunStatus.SUCCESS
        sent = _llm_messages(follow, 0)
        assert sent[0].images or "not shown" in sent[0].content
        assert sent[0].content.startswith("see")
        state = follow.loop.seen[-1]["state"]  # type: ignore[union-attr]
        assert state.sent_through == 1
        assert [e.owner_kind for e in store.edges_of(result.run_id)] == ["seed"]

    async def test_text_mode_replay_uses_input_parts(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="one")])
        source = await agent.run(["see", Image.from_bytes(PNG)])
        follow = _agent(store, [LLMResponse(text="two")])
        await follow.run("and?", runs=[source.run_id], replay_last=0)
        sent = _llm_messages(follow, 0)
        assert sent[0].images and sent[1].content == "one"

    async def test_new_image_on_text_only_model_raises(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="one")])
        source = await agent.run(["see", Image.from_bytes(PNG)])
        follow = _agent(
            store,
            [LLMResponse(text="two")],
            images=ImagePolicy(capability_overrides={"mock": "no"}),
        )
        before = await _count_runs(store)
        with pytest.raises(UnsupportedModalityError):
            await follow.run(["new", Image.from_bytes(PNG2)], runs=[source.run_id])
        assert await _count_runs(store) == before


# ---------------------------------------------------------------------------
# OTel image attributes
# ---------------------------------------------------------------------------


class TestOtelImages:
    async def test_llm_span_gets_image_attributes(self) -> None:
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from dendrux.notifiers.otel import OpenTelemetryNotifier

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("t"))
        ref = ImageRef(
            sha256="d" * 64, media_type="image/png", byte_length=10, width=400, height=200
        )
        await notifier.on_run_started("r", agent_name="a", agent_model="claude-sonnet-4-6")
        await notifier.on_llm_call_started("r", 1)
        await notifier.on_llm_call_completed(
            "r",
            LLMResponse(text="x"),
            1,
            semantic_messages=[Message.with_parts(Role.USER, [TextPart("q"), ImagePart(ref)])],
        )
        span = next(s for s in exporter.get_finished_spans() if "chat" in s.name)
        assert span.attributes["dendrux.images.count"] == 1
        assert span.attributes["dendrux.images.estimated_tokens"] == 120

    async def test_text_only_span_has_no_image_attributes(self) -> None:
        pytest.importorskip("opentelemetry.sdk")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from dendrux.notifiers.otel import OpenTelemetryNotifier

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        notifier = OpenTelemetryNotifier(tracer=provider.get_tracer("t"))
        await notifier.on_run_started("r", agent_name="a", agent_model="gpt-5-mini")
        await notifier.on_llm_call_started("r", 1)
        await notifier.on_llm_call_completed(
            "r", LLMResponse(text="x"), 1, semantic_messages=[Message(role=Role.USER, content="q")]
        )
        span = next(s for s in exporter.get_finished_spans() if "chat" in s.name)
        assert "dendrux.images.count" not in (span.attributes or {})


# ---------------------------------------------------------------------------
# Context folding with parts (U-03). Kept here: tests/unit/test_context_blocks*.py
# are whole-file R-04 protected by the v0.3.0 oracle manifest.
# ---------------------------------------------------------------------------


class TestWireUserParts:
    def _img(self, sha: str = "e") -> ImagePart:
        return ImagePart(
            ImageRef(sha256=sha * 64, media_type="image/png", byte_length=3, width=1, height=1)
        )

    def test_no_context_is_identity(self) -> None:
        from dendrux.context_blocks import wire_user_parts

        parts = (TextPart("a"), self._img(), TextPart("b"))
        assert wire_user_parts(parts, [], None) == parts
        image_only = (self._img(), self._img("f"))
        assert wire_user_parts(image_only, [], None) == image_only

    def test_leading_text_is_framed(self) -> None:
        from dendrux.context_blocks import wire_user_input, wire_user_parts

        ctx = [ContextBlock("S", placement="stable"), ContextBlock("D")]
        parts = (TextPart("q"), self._img(), TextPart("tail"))
        out = wire_user_parts(parts, ["ENV"], ctx)
        assert out[0] == TextPart(wire_user_input("q", ["ENV"], ctx))
        assert out[1:] == parts[1:]

    def test_leading_image_gets_framing_part(self) -> None:
        from dendrux.context_blocks import wire_user_input, wire_user_parts

        ctx = [ContextBlock("S", placement="stable")]
        parts = (self._img(), TextPart("q"))
        out = wire_user_parts(parts, [], ctx)
        assert out[0] == TextPart(wire_user_input("", [], ctx))
        assert out[1:] == parts

    def test_stable_band_bytes_do_not_depend_on_images(self) -> None:
        from dendrux.context_blocks import wire_user_parts

        ctx = [ContextBlock("STABLE BYTES", placement="stable")]
        a = wire_user_parts((TextPart("q"), self._img()), [], ctx)
        b = wire_user_parts((TextPart("q"), self._img("f")), [], ctx)
        assert a[0] == b[0]

    def test_fold_context_without_history_returns_parts_message(self) -> None:
        from dendrux.context_blocks import fold_context

        parts = (TextPart("q"), self._img())
        out = fold_context([], None, "q", user_parts=parts)
        assert out is not None and len(out) == 1
        assert out[0].parts == parts and out[0].placement == "dynamic"
        stable = fold_context([], [ContextBlock("S", placement="stable")], "q", user_parts=parts)
        assert stable is not None and stable[0].placement == "stable"

    def test_fold_context_folds_stable_into_first_parts_message(self) -> None:
        from dendrux.context_blocks import fold_context

        first = Message.with_parts(Role.USER, [self._img(), TextPart("earlier")])
        history = [first, Message(role=Role.ASSISTANT, content="ok")]
        out = fold_context(history, [ContextBlock("S", placement="stable")], "now")
        assert out is not None
        assert out[0].placement == "stable"
        assert out[0].parts[0] == TextPart("STABLE CONTEXT:\nS\n\nUSER MESSAGE:\n")
        assert out[0].parts[1:] == first.parts
        assert out[-1].content == "now" and out[-1].parts is None

    def test_text_only_fold_unchanged(self) -> None:
        from dendrux.context_blocks import fold_context

        history = [Message(role=Role.USER, content="Q1"), Message(role=Role.ASSISTANT, content="A")]
        ctx = [ContextBlock("S", placement="stable")]
        out = fold_context(history, ctx, "now")
        assert out is not None and out[0].parts is None
        assert out[0].content == "STABLE CONTEXT:\nS\n\nUSER MESSAGE:\nQ1"
        assert fold_context([], None, "q") is None


class TestCapability:
    class _Provider:
        def __init__(self, support: str, learned: str | None = None, source: str = "adapter"):
            self.support, self.learned, self.source = support, learned, source

        def learned_image_support(self, model: str) -> str | None:
            return self.learned

        async def image_input_support(self, model: str) -> str:
            return self.support

        def image_capability_source(self) -> str:
            return self.source

    async def test_precedence(self) -> None:
        from dendrux.runtime.submit import effective_image_capability as eff

        p = self._Provider("yes", learned="no")
        assert await eff(p, "m", ImagePolicy()) == "no"  # type: ignore[arg-type]
        assert await eff(p, "m", ImagePolicy(capability_overrides={"m": "yes"})) == "yes"  # type: ignore[arg-type]
        u = self._Provider("unknown")
        assert await eff(u, "m", ImagePolicy()) == "unknown"  # type: ignore[arg-type]
        assert await eff(u, "m", ImagePolicy(unknown_capability="error")) == "no"  # type: ignore[arg-type]

    async def test_error_source_from_provider(self) -> None:
        from dendrux.runtime.submit import check_new_image_capability as check

        with pytest.raises(UnsupportedModalityError) as info:
            await check(
                self._Provider("no", source="catalog"),  # type: ignore[arg-type]
                "m",
                ImagePolicy(),
                image_count=2,
                origin="user",
            )
        assert (info.value.source, info.value.image_count) == ("catalog", 2)
        with pytest.raises(UnsupportedModalityError) as info:
            await check(
                self._Provider("yes", source="catalog"),  # type: ignore[arg-type]
                "m",
                ImagePolicy(capability_overrides={"m": "no"}),
                image_count=1,
                origin="user",
            )
        assert info.value.source == "capability"
        await check(
            self._Provider("unknown"),  # type: ignore[arg-type]
            "m",
            ImagePolicy(),
            image_count=1,
            origin="user",
        )


class TestD1Seams:
    def test_stream_error_meta_lists_rejected_images(self) -> None:
        from dendrux.errors import ImageRejectedError
        from dendrux.runtime.runner import _stream_error_meta

        exc = ImageRejectedError(provider="p", model="m", image_ids=("img_a", "img_b"))
        meta = _stream_error_meta({"mcp": 1}, exc)
        assert meta == {
            "mcp": 1,
            "error_type": "ImageRejectedError",
            "image_rejected": ["img_a", "img_b"],
        }
        assert "image_rejected" not in _stream_error_meta({}, ValueError("x"))

    def test_image_state_event_scope_filter(self) -> None:
        src = ImageSource(
            "t:a",
            [
                _event(
                    "image.state",
                    {"tenant_scope": "t:b", "described": [{"sha256": "x", "text": "b"}]},
                    1,
                ),
                _event(
                    "image.state",
                    {"tenant_scope": "t:a", "described": [{"sha256": "x", "text": "a"}]},
                    2,
                ),
            ],
        )
        state = initial_image_view(
            sent_through=0, sources=[src], tenant_scope="t:a", single_full=True
        )
        assert state.described == {("t:a", "x"): "a"}


# ---------------------------------------------------------------------------
# Review fix round (review-D2.md)
# ---------------------------------------------------------------------------

JPG = (FIXTURES / "zebra42.jpg").read_bytes()
GIF = (FIXTURES / "zebra42.gif").read_bytes()
SHOTS = [PNG, PNG2, JPG, GIF]
CORRUPT = PNG[:120]


@tool()
async def shot(n: int) -> Any:
    """Server screenshot."""
    from dendrux.media import ToolOutput

    return ToolOutput(f"shot {n}", images=[Image.from_bytes(SHOTS[n])])


def _visible_tools(msgs: list[Message]) -> list[str]:
    return [m.content.split("\n")[0] for m in msgs if m.role == Role.TOOL and m.images]


class TestSeededSourceCarry:
    """P1-1: a single-source carry is shifted by the source's own seeded slots."""

    async def _scenario(self, store: ImageStore, history: list[ChatMessage]) -> tuple[Any, Any]:
        policy = ImagePolicy(
            drop=ImageDropPolicy(keep_recent=2, interval=2), capability_overrides={"mock": "yes"}
        )
        calls = [
            LLMResponse(text=None, tool_calls=[ToolCall(name="shot", params={"n": i})])
            for i in (1, 2, 3, 0, 1)
        ]
        src_agent = _agent(store, [*calls, LLMResponse(text="done")], tools=[shot], images=policy)
        source = await src_agent.run("go", history=history)
        follow = _agent(store, [LLMResponse(text="next")], tools=[shot], images=policy)
        await follow.run("more", runs=[source.run_id], replay_last=1)
        return _llm_messages(src_agent, -1), _llm_messages(follow, 0)

    async def test_control_and_seeded_source_show_the_same_images(self, store: ImageStore) -> None:
        src_c, fol_c = await self._scenario(
            store, [ChatMessage.user("old"), ChatMessage.assistant("ok")]
        )
        assert _visible_tools(fol_c) == _visible_tools(src_c)
        src_h, fol_h = await self._scenario(
            store, [ChatMessage.user(["old", Image.from_bytes(PNG)]), ChatMessage.assistant("ok")]
        )
        assert _visible_tools(src_h) == _visible_tools(src_c)
        assert _visible_tools(fol_h) == _visible_tools(src_h)
        dropped_src = [m.content for m in src_h if "omitted from context" in m.content]
        dropped_fol = [m.content for m in fol_h if "omitted from context" in m.content]
        assert dropped_src and dropped_fol == dropped_src

    def test_shift_applies_to_view_and_every_placeholder(self) -> None:
        src = ImageSource(
            "-",
            [
                _event("image.view_changed", {"epoch": 3, "dropped_through": 6}, 1),
                _event(
                    "image.state",
                    {
                        "placeholders": [
                            {"slot": 0, "reason": "policy"},  # the source's seeded slot
                            {"slot": 2, "reason": "rejected"},
                            {"slot": 3, "reason": "purged"},
                            {"slot": 4, "reason": "drop"},
                        ]
                    },
                    2,
                ),
            ],
            slot_offset=2,
        )
        state = initial_image_view(
            sent_through=9, sources=[src], tenant_scope="-", single_full=True
        )
        assert (state.epoch, state.dropped_through) == (3, 4)
        assert state.placeholders == {0: "rejected", 1: "purged", 2: "drop"}

    def test_unknown_offset_starts_a_fresh_view(self) -> None:
        src = ImageSource(
            "-",
            [
                _event("image.view_changed", {"epoch": 3, "dropped_through": 6}, 1),
                _event("image.rejected", {"sha256": "r"}, 2),
                _event("image.state", {"placeholders": [{"slot": 0, "reason": "drop"}]}, 3),
            ],
            slot_offset=None,
        )
        state = initial_image_view(
            sent_through=1, sources=[src], tenant_scope="-", single_full=True
        )
        assert (state.epoch, state.dropped_through, state.placeholders) == (0, 0, {})
        assert state.rejected == [("-", "r")]  # key-based state still carries

    def test_source_offset_from_meta(self) -> None:
        from dendrux.runtime.replay import IMAGE_SEEDED_KEY, PREFIX_META_KEY, source_slot_offset

        assert source_slot_offset(None) is None
        assert source_slot_offset({PREFIX_META_KEY: {}}) == 0
        assert source_slot_offset({PREFIX_META_KEY: {}, IMAGE_SEEDED_KEY: 3}) == 3
        assert source_slot_offset({PREFIX_META_KEY: {}, IMAGE_SEEDED_KEY: None}) is None

    async def test_seeded_slots_recorded_in_run_meta(self, store: ImageStore) -> None:
        from dendrux.runtime.replay import IMAGE_SEEDED_KEY

        agent = _agent(store, [LLMResponse(text="a"), LLMResponse(text="b")])
        history = [ChatMessage.user(["old", Image.from_bytes(PNG)]), ChatMessage.assistant("ok")]
        seeded = await agent.run("q", history=history)
        plain = await agent.run(["q", Image.from_bytes(PNG2)])
        assert (await store.get_run(seeded.run_id)).meta[IMAGE_SEEDED_KEY] == 1
        assert IMAGE_SEEDED_KEY not in (await store.get_run(plain.run_id)).meta


class TestOnePhaseOne:
    """P2-2: every image is validated before any blob is written."""

    async def test_second_tool_result_invalid_writes_nothing(self, store: ImageStore) -> None:
        responses = [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(name="take_screenshot", params={}),
                    ToolCall(name="take_screenshot", params={}),
                ],
            )
        ]
        agent = _agent(store, responses)
        paused = await agent.run("two shots", tenant_id="acme")
        ids = [
            tc["id"] for tc in (await store.get_pause_state(paused.run_id))["pending_tool_calls"]
        ]
        with pytest.raises(InvalidImageError):
            await _agent(store, []).submit_tool_results(
                paused.run_id,
                [
                    ToolResult("take_screenshot", ids[0], '"a"', images=(Image.from_bytes(PNG),)),
                    ToolResult(
                        "take_screenshot", ids[1], '"b"', images=(Image.from_bytes(CORRUPT),)
                    ),
                ],
            )
        assert store.rows == {}
        assert (await store.get_run(paused.run_id)).status == RunStatus.WAITING_CLIENT_TOOL.value

    async def test_valid_input_with_invalid_history_writes_nothing(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="x")])
        history = [ChatMessage.user([Image.from_bytes(CORRUPT)]), ChatMessage.assistant("ok")]
        with pytest.raises(InvalidImageError):
            await agent.run([Image.from_bytes(PNG)], history=history)
        assert store.rows == {} and await _count_runs(store) == 0

    async def test_capability_runs_once_after_all_validation(
        self, store: ImageStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dendrux.runtime.submit as submit

        order: list[str] = []
        real = submit.check_new_image_capability

        async def spy(*args: Any, **kwargs: Any) -> None:
            order.append(f"capability:{kwargs['image_count']}:{len(store.rows)}")
            await real(*args, **kwargs)

        monkeypatch.setattr(submit, "check_new_image_capability", spy)
        agent = _agent(store, [LLMResponse(text="x")])
        history = [ChatMessage.user([Image.from_bytes(PNG2)]), ChatMessage.assistant("ok")]
        await agent.run([Image.from_bytes(PNG)], history=history)
        assert order == ["capability:1:0"]  # input images only, before any row
        assert len(store.rows) == 2


class TestResumeStreamValidatesFirst:
    """P2-3: resume_stream checks the run before it ingests anything."""

    async def test_wrong_status_and_call_id_write_no_blob(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="x")])
        run_id = await _human_pause(store, agent)
        stream = agent.resume_stream(
            run_id,
            tool_results=[
                ToolResult("take_screenshot", "nope", '"x"', images=(Image.from_bytes(PNG),))
            ],
        )
        with pytest.raises(ValueError):
            async for _ in stream:
                pass
        assert store.rows == {}
        assert (await store.get_run(run_id)).status == RunStatus.WAITING_HUMAN_INPUT.value

        paused = await _agent(store, _screenshot_responses()).run("shot")
        stream = _agent(store, []).resume_stream(
            paused.run_id,
            tool_results=[
                ToolResult("take_screenshot", "wrong", '"x"', images=(Image.from_bytes(PNG),))
            ],
        )
        with pytest.raises(ValueError, match="call ids"):
            async for _ in stream:
                pass
        assert store.rows == {}

    async def test_missing_run_raises_not_found(self, store: ImageStore) -> None:
        from dendrux.errors import RunNotFoundError

        stream = _agent(store, []).resume_stream(
            "01NOSUCHRUN000000000000000", user_input=[Image.from_bytes(PNG)]
        )
        with pytest.raises(RunNotFoundError):
            async for _ in stream:
                pass
        assert store.rows == {}

    async def test_resume_stream_checks_run_stream_signature(self, store: ImageStore) -> None:
        class RunOnlyLoop(LegacyLoop):
            async def run(self, **kwargs: Any) -> Any:  # accepts **kwargs (image_runtime too)
                return await ReActLoop.run(self, **kwargs)

        paused = await _agent(store, _screenshot_responses()).run("shot")
        call_id = (await store.get_pause_state(paused.run_id))["pending_tool_calls"][0]["id"]
        agent = _agent(store, [], loop=RunOnlyLoop())
        stream = agent.resume_stream(
            paused.run_id,
            tool_results=[
                ToolResult("take_screenshot", call_id, '"x"', images=(Image.from_bytes(PNG),))
            ],
        )
        with pytest.raises(TypeError, match="run_stream"):
            async for _ in stream:
                pass


class TestCustomLoopReplay:
    """P2-4: a loop that cannot take image_runtime refuses an image prefix before create_run."""

    async def test_runs_replay_of_image_run(self, store: ImageStore) -> None:
        source = await _agent(store, [LLMResponse(text="x")]).run(["look", Image.from_bytes(PNG)])
        before = await _count_runs(store)
        agent = _agent(store, [LLMResponse(text="y")], loop=LegacyLoop())
        with pytest.raises(TypeError, match="image_runtime"):
            await agent.run("next", runs=[source.run_id], replay_last="all")
        with pytest.raises(TypeError, match="image_runtime"):
            await agent.retry(source.run_id)
        assert await _count_runs(store) == before
        events = [e async for e in agent.stream("next", runs=[source.run_id], replay_last=1)]
        assert events[-1].run_result.meta["error_type"] == "TypeError"
        assert await _count_runs(store) == before

    async def test_text_replay_with_legacy_loop_still_works(self, store: ImageStore) -> None:
        source = await _agent(store, [LLMResponse(text="x")]).run("text only")
        agent = _agent(store, [LLMResponse(text="y")], loop=LegacyLoop())
        result = await agent.run("next", runs=[source.run_id], replay_last="all")
        assert result.status == RunStatus.SUCCESS


class TestGarbageCollectionWiring:
    """P2-5: a full blob store triggers one collect_garbage(older_than=1h) pass."""

    async def test_runtime_gets_one_hour_gc_pass(self, store: ImageStore) -> None:
        from datetime import timedelta

        from dendrux.media import InMemoryBlobStore

        calls: list[timedelta] = []
        blob_store = InMemoryBlobStore(max_bytes=len(PNG) + 16)

        class _Accessor:
            async def collect_garbage(self, *, older_than: timedelta, limit: int = 1000) -> Any:
                calls.append(older_than)
                for name in blob_store.object_names():  # simulate GC freeing old objects
                    scope, sha, gen = name.split("/")
                    from urllib.parse import unquote

                    from dendrux.media import BlobKey

                    await blob_store.delete(BlobKey(unquote(scope), sha, gen))
                return None

        agent = _agent(store, [LLMResponse(text="a"), LLMResponse(text="b")], blob_store=blob_store)
        agent._blob_services(store).accessor = _Accessor()
        await agent.run([Image.from_bytes(PNG)])
        assert calls == []
        result = await agent.run([Image.from_bytes(PNG2)])  # store full -> one GC pass
        assert result.status == RunStatus.SUCCESS
        assert calls == [timedelta(hours=1)]
        assert agent.loop.seen[-1]["runtime"].collect_garbage is not None  # type: ignore[union-attr]


class TestIdempotencyAfterPurge:
    """P2-7: a keyed retry whose image was purged returns the existing run."""

    async def test_keyed_retry_after_purge_returns_existing(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="a")])
        first = await agent.run(["q", Image.from_bytes(PNG)], idempotency_key="kp", tenant_id="t")
        await store.mark_purging("t:t", _sha(PNG), "gdpr")
        again = await agent.run(["q", Image.from_bytes(PNG)], idempotency_key="kp", tenant_id="t")
        assert again.run_id == first.run_id and again.answer == "a"
        with pytest.raises(IdempotencyConflictError):
            await agent.run(["other", Image.from_bytes(PNG)], idempotency_key="kp", tenant_id="t")
        with pytest.raises(InvalidImageError, match="purged"):
            await agent.run(["q", Image.from_bytes(PNG)], tenant_id="t")

    async def test_keyed_stream_after_purge_yields_cached(self, store: ImageStore) -> None:
        agent = _agent(store, [LLMResponse(text="a")])
        first = await agent.run([Image.from_bytes(PNG)], idempotency_key="ks")
        await store.mark_purging("-", _sha(PNG), None)
        events = [e async for e in agent.stream([Image.from_bytes(PNG)], idempotency_key="ks")]
        assert len(events) == 1 and events[0].run_result.run_id == first.run_id


class TestSubmitEdgeKeyRace:
    """P2-7: the claim re-checks the pause iteration that keyed the submission edges."""

    async def test_repaused_run_rejects_stale_submission(
        self, store: ImageStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dendrux.errors import RunAlreadyClaimedError

        agent = _agent(store, [LLMResponse(text="thanks")])
        run_id = await _human_pause(store, agent)
        real = store.get_pause_state

        async def stale(rid: str) -> Any:
            data = await real(rid)
            return {**data, "iteration": data["iteration"] - 1} if data else data

        monkeypatch.setattr(store, "get_pause_state", stale)
        with pytest.raises(RunAlreadyClaimedError):
            await agent.submit_input(run_id, ["here", Image.from_bytes(PNG)])
        monkeypatch.undo()
        assert (await store.get_run(run_id)).status == RunStatus.WAITING_HUMAN_INPUT.value
        assert store.edges_of(run_id) == []


class TestStoreFullSinglePass:
    """A full blob store runs exactly one 1h GC pass before failing (§5.12)."""

    @staticmethod
    def _real_services(monkeypatch: pytest.MonkeyPatch) -> None:
        from dendrux.runtime._blob_catalog import BlobServices

        monkeypatch.setattr("dendrux.agent._blob_services_factory", lambda: BlobServices)

    @staticmethod
    def _spy(services: Any) -> list[Any]:
        calls: list[Any] = []
        real = services.accessor.collect_garbage

        async def spy(**kwargs: Any) -> Any:
            calls.append(kwargs.get("older_than"))
            return await real(**kwargs)

        services.accessor.collect_garbage = spy
        return calls

    async def test_no_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import timedelta

        from dendrux.media import InMemoryBlobStore

        self._real_services(monkeypatch)
        agent = Agent(
            prompt="p",
            provider=MockLLM([LLMResponse(text="a"), LLMResponse(text="b")]),
            loop=ImageLoop(),
            images=ImagePolicy(capability_overrides={"mock": "yes"}),
            blob_store=InMemoryBlobStore(max_bytes=len(PNG) + 16),
        )
        await agent.run([Image.from_bytes(PNG)])
        assert agent.loop.seen[-1]["runtime"].collect_garbage is None  # type: ignore[union-attr]
        calls = self._spy(agent._blob_services(None))
        with pytest.raises(ImageLimitError, match="store_bytes"):
            await agent.run([Image.from_bytes(PNG2)])
        assert calls == [timedelta(hours=1)]  # the store wrapper's single pass

    async def test_database_backed(self, engine, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import timedelta

        from dendrux.media import InMemoryBlobStore

        self._real_services(monkeypatch)
        store = SQLAlchemyStateStore(engine)
        agent = _agent(
            store,
            [LLMResponse(text="a"), LLMResponse(text="b")],
            blob_store=InMemoryBlobStore(max_bytes=len(PNG) + 16),
        )
        await agent.run([Image.from_bytes(PNG)])
        assert agent.loop.seen[-1]["runtime"].collect_garbage is not None  # type: ignore[union-attr]
        calls = self._spy(agent._blob_services(store))
        before = await store.count_runs()
        with pytest.raises(ImageLimitError, match="store_bytes"):
            await agent.run([Image.from_bytes(PNG2)])
        assert calls == [timedelta(hours=1)]  # ingest's single pass
        assert await store.count_runs() == before
