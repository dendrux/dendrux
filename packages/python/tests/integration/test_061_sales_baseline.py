"""Normal sales cleanup and whole-batch approval on SQLite and Postgres."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dendrux import Agent, Budget
from dendrux.guardrails import SecretDetection
from dendrux.tool import get_tool_def
from dendrux.types import Role, RunStatus
from tests._helpers.sales_061 import SalesFixture, SalesProvider, sales_sheet

if TYPE_CHECKING:
    from dendrux.runtime.state import StateStore

__all__ = ["sales_sheet"]


def test_write_region_schema_enum(sales_sheet: SalesFixture) -> None:
    schema = get_tool_def(sales_sheet.sheet.write_region).parameters
    assert schema["properties"]["region"].get("enum") == ["NORTH", "SOUTH"]


@pytest.mark.parametrize("engine", ["sqlite_file", "postgres"], indirect=True)
@pytest.mark.parametrize("approved", [True, False], ids=["approve", "decline"])
async def test_sales_cleanup(store: StateStore, sales_sheet: SalesFixture, approved: bool) -> None:
    provider = SalesProvider(approved=approved)
    original = await sales_sheet.sheet.read_sheet()
    async with Agent(
        name="SalesCleaner",
        provider=provider,
        state_store=store,
        prompt="Read the sheet, trim and uppercase each region (NORTH or SOUTH), "
        "then post a review notice. "
        "Stop if the batch is declined.",
        tools=sales_sheet.tools,
        require_approval=["write_region"],
        guardrails=[SecretDetection()],
        max_iterations=8,
        budget=Budget(max_tokens=20000, on_exceed="cancel"),
    ) as agent:
        paused = await agent.run(
            "Clean the sales regions and post a review notice.", tenant_id="sales-demo"
        )
        assert paused.status == RunStatus.WAITING_APPROVAL
        assert paused.pending is not None and paused.pause_id is not None
        assert [call.name for call in paused.pending.tool_calls] == [
            "write_region",
            "post_review_notice",
        ]
        assert sales_sheet.ledger.entries() == []
        assert await sales_sheet.sheet.read_sheet() == original
        assert not sales_sheet.sheet.inbox.exists()
        result = await agent.submit_approval(
            paused.run_id,
            approved=approved,
            tenant_id="sales-demo",
            expected_pause_id=paused.pause_id,
        )

    assert result.status == RunStatus.SUCCESS
    assert result.run_id == paused.run_id
    assert result.answer == provider.final_answer
    assert provider.calls == provider.closed_streams == 3
    rows = await sales_sheet.sheet.read_sheet()
    assert [(row["order"], row["amount"]) for row in rows] == [
        ("A1", "100"),
        ("A2", "200"),
        ("A3", "50"),
    ]
    observations = [
        json.loads(message.content) for message in provider.messages[2] if message.role == Role.TOOL
    ][-2:]
    if approved:
        assert [row["region"] for row in rows] == ["NORTH", "SOUTH", "NORTH"]
        assert sales_sheet.ledger.entries() == ["write_region", "post_review_notice"]
        assert sales_sheet.sheet.inbox.read_text() == "Sales cleanup ready for review\n"
        assert observations == ["region updated", "notice posted"]
    else:
        assert rows == original
        assert sales_sheet.ledger.entries() == []
        assert not sales_sheet.sheet.inbox.exists()
        assert observations == ["User declined to run this tool."] * 2


@pytest.mark.parametrize("row_number,region", [(0, "NORTH"), (4, "SOUTH"), (1, "WEST")])
async def test_invalid_cell_preserves_sheet(
    sales_sheet: SalesFixture, row_number: int, region: str
) -> None:
    original = sales_sheet.sheet.path.read_bytes()
    with pytest.raises(ValueError, match="demo_invalid_cell_write"):
        await sales_sheet.sheet.write_region(row_number, region)
    assert sales_sheet.sheet.path.read_bytes() == original


@pytest.mark.parametrize("count", [30, 100, 101])
async def test_sheet_row_bound(sales_sheet: SalesFixture, count: int) -> None:
    sales_sheet.sheet.path.write_text("order,region,amount\n" + "A1,NORTH,100\n" * count)
    if count > 100:
        with pytest.raises(ValueError, match="demo_sheet_too_many_rows"):
            await sales_sheet.sheet.read_sheet()
    else:
        assert len(await sales_sheet.sheet.read_sheet()) == count


@pytest.mark.parametrize("size", [32768, 32769])
async def test_sheet_byte_bound(sales_sheet: SalesFixture, size: int) -> None:
    prefix, suffix = b"order,region,amount\n", b",NORTH,100\n"
    padding = size - len(prefix + suffix)
    order = b"\xc3\xa9" * (padding // 2) + b"A" * (padding % 2)
    sales_sheet.sheet.path.write_bytes(prefix + order + suffix)
    if size > 32768:
        with pytest.raises(ValueError, match="demo_sheet_too_large"):
            await sales_sheet.sheet.read_sheet()
    else:
        assert (await sales_sheet.sheet.read_sheet())[0]["amount"] == "100"


@pytest.mark.parametrize(
    "contents",
    [
        "region,amount\nNORTH,100\n",
        "order,region,amount\nA1,NORTH\n",
        "order,region,amount\nA1,NORTH,100,extra\n",
    ],
)
async def test_malformed_sheet_preserved(sales_sheet: SalesFixture, contents: str) -> None:
    sales_sheet.sheet.path.write_text(contents)
    with pytest.raises(ValueError, match="demo_invalid_sheet"):
        await sales_sheet.sheet.write_region(1, "SOUTH")
    assert sales_sheet.sheet.path.read_text() == contents


async def test_replace_failure_preserves_sheet(
    sales_sheet: SalesFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = sales_sheet.sheet.path.read_bytes()

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("test replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="test replace failure"):
        await sales_sheet.sheet.write_region(1, "NORTH")
    assert sales_sheet.sheet.path.read_bytes() == original
    assert list(sales_sheet.sheet.path.parent.glob("*.tmp")) == []


async def test_initialize_preserves_edits_and_notice_appends(sales_sheet: SalesFixture) -> None:
    await sales_sheet.sheet.write_region(1, "SOUTH")
    sales_sheet.sheet.initialize()
    assert (await sales_sheet.sheet.read_sheet())[0]["region"] == "SOUTH"
    await sales_sheet.sheet.post_review_notice()
    await sales_sheet.sheet.post_review_notice()
    assert sales_sheet.sheet.inbox.read_text().splitlines() == [
        "Sales cleanup ready for review",
        "Sales cleanup ready for review",
    ]


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "decline"])
async def test_example_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    approved: bool,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    example = importlib.import_module("41_clean_sales")
    provider = SalesProvider(approved=approved)
    monkeypatch.setattr(example, "AnthropicProvider", lambda **kwargs: provider)
    monkeypatch.setenv("SALES_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("DENDRUX_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}")
    monkeypatch.setattr("builtins.input", lambda prompt: "y" if approved else "n")
    await example.main()
    assert provider.final_answer in capsys.readouterr().out
    assert provider.calls == provider.closed_streams == 3
    assert (tmp_path / "review-inbox.log").exists() == approved
