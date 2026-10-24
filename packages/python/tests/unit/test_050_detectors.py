"""0.5.0 T3: built-in detector quality, linear-time scans and off-loop scanning."""

from __future__ import annotations

import asyncio
import time

import pytest

from dendrux.agent import Agent
from dendrux.guardrails import PII, Pattern, PromptInjection, SecretDetection
from dendrux.llm.mock import MockLLM
from dendrux.loops.react import ReActLoop
from dendrux.strategies.native import NativeToolCalling
from dendrux.types import LLMResponse, RunStatus, UsageStats

_EMAIL = "alice@example.com"


# ---------------------------------------------------------------------------
# D-028: detector quality (labelled corpus)
# ---------------------------------------------------------------------------

_PII_CORPUS: list[tuple[str, set[str]]] = [
    # positives
    ("Contact jane.doe+x@example.co.uk now", {"EMAIL_ADDRESS"}),
    ("Call +1-555-123-4567", {"PHONE_NUMBER"}),
    ("Call +442079460958", {"PHONE_NUMBER"}),
    ("Call (212) 555-0147 today", {"PHONE_NUMBER"}),
    ("Call 212-555-0147 today", {"PHONE_NUMBER"}),
    ("Call 212.555.0147 today", {"PHONE_NUMBER"}),
    ("Call 1-800-555-0199 today", {"PHONE_NUMBER"}),
    ("phone: 2125550147", {"PHONE_NUMBER"}),
    ("Mobile 2125550147", {"PHONE_NUMBER"}),
    ("SSN 123-45-6789", {"US_SSN"}),
    ("card 4111111111111111", {"CREDIT_CARD"}),
    ("card 4111 1111 1111 1111", {"CREDIT_CARD"}),
    ("card 4111-1111-1111-1111", {"CREDIT_CARD"}),
    ("card 5555555555554444", {"CREDIT_CARD"}),
    ("Host 192.168.1.1.", {"IP_ADDRESS"}),
    ("Host 10.0.0.255 up", {"IP_ADDRESS"}),
    ("Host 255.255.255.255", {"IP_ADDRESS"}),
    # negatives: the false positives this release removes
    ("order 1234567890 shipped", set()),
    ("tracking 9400111899223197428490", set()),
    ("timestamp 1699999999999", set()),
    ("id 20240115123456", set()),
    ("card 4111-1111-1111-1112", set()),
    ("number 1234567812345678", set()),
    ("ip 999.1.1.1", set()),
    ("ip 256.256.256.256", set()),
    ("version 1.2.3.4.5", set()),
    ("ip 300.1.1.1", set()),
    ("date 2024-10-24 and 12:30:45", set()),
    ("user@localhost", set()),
    ("a@b.c", set()),
]

_SECRET_CORPUS: list[tuple[str, set[str]]] = [
    ("Key: AKIAIOSFODNN7EXAMPLE", {"AWS_ACCESS_KEY"}),
    (
        "AKIAIOSFODNN7EXAMPLE wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        {"AWS_ACCESS_KEY", "AWS_SECRET_KEY"},
    ),
    ("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", {"AWS_SECRET_KEY"}),
    ("api_key=sk_live_" + "ABCDEFGHIJKLMNOPQRSTUVWX1234", {"GENERIC_API_KEY"}),
    ("-----BEGIN RSA PRIVATE KEY-----", {"PRIVATE_KEY"}),
    # negatives
    ("commit da39a3ee5e6b4b0d3255bfef95601890afd80709 fixes it", set()),
    ("sha1 da39a3ee5e6b4b0d3255bfef95601890afd80709", set()),
    ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", set()),
    ("hello world", set()),
]


class TestDetectorCorpus:
    @pytest.mark.parametrize(("text", "expected"), _PII_CORPUS)
    async def test_pii(self, text: str, expected: set[str]) -> None:
        found = {f.entity_type for f in await PII().scan(text)}
        assert found == expected

    @pytest.mark.parametrize(("text", "expected"), _SECRET_CORPUS)
    async def test_secrets(self, text: str, expected: set[str]) -> None:
        found = {f.entity_type for f in await SecretDetection().scan(text)}
        assert found == expected

    async def test_git_sha_in_prompt_does_not_block_a_run(self) -> None:
        agent = Agent(prompt="p", tools=[], guardrails=[SecretDetection(action="block")])
        llm = MockLLM([LLMResponse(text="ok", usage=UsageStats(input_tokens=1, output_tokens=1))])
        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="my commit is da39a3ee5e6b4b0d3255bfef95601890afd80709 please summarize",
        )
        assert result.status == RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# D-056: linear time and off-loop scans
# ---------------------------------------------------------------------------

_ADVERSARIAL_UNITS = [
    "a",
    "1",
    "1.",
    "a@",
    "a" * 60 + "@",
    "a.b-c",
    " ",
    "+",
    "(",
    "token ",
    "AKIA",
    "A/",
    "1 ",
    "1.1.1.",
    "1111 ",
    "da39a3ee5e6b4b0d3255bfef95601890afd80709 ",
]


def _best_scan_time(pii: PII, sec: SecretDetection, text: str) -> float:
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        asyncio.run(_scan_both(pii, sec, text))
        best = min(best, time.perf_counter() - start)
    return best


async def _scan_both(pii: PII, sec: SecretDetection, text: str) -> None:
    await pii.scan(text)
    await sec.scan(text)


class TestScanScaling:
    @pytest.mark.parametrize("unit", _ADVERSARIAL_UNITS)
    def test_ten_times_the_text_costs_under_fifteen_times(self, unit: str) -> None:
        pii, sec = PII(), SecretDetection()

        def make(n: int) -> str:
            return (unit * (n // len(unit) + 1))[:n]

        small = _best_scan_time(pii, sec, make(100_000))
        large = _best_scan_time(pii, sec, make(1_000_000))
        assert large / max(small, 1e-4) < 15, (unit, small, large)

    def test_trailing_at_sign_does_not_go_quadratic(self) -> None:
        pii, sec = PII(), SecretDetection()
        small = _best_scan_time(pii, sec, "a" * 100_000 + "@")
        large = _best_scan_time(pii, sec, "a" * 1_000_000 + "@")
        assert large / max(small, 1e-4) < 15


class TestScansLeaveTheLoop:
    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        calls: list[str] = []
        real = asyncio.to_thread

        async def spy(fn, /, *args, **kwargs):
            calls.append(getattr(fn, "__name__", "?"))
            return await real(fn, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", spy)
        return calls

    async def test_large_text_scans_in_a_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        text = ("x" * 99 + " ") * 400 + _EMAIL  # about 40 KB
        findings = await PII().scan(text)
        assert [f.entity_type for f in findings] == ["EMAIL_ADDRESS"]
        assert calls == ["_scan_sync"]

    async def test_small_text_stays_on_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        await PII().scan(f"mail {_EMAIL}")
        await SecretDetection().scan("hello")
        assert calls == []

    async def test_other_built_in_guardrails_offload_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._spy(monkeypatch)
        big = "y " * 20_000
        await SecretDetection().scan(big)
        await PromptInjection(patterns=[Pattern("X", r"zzz")]).scan(big)
        assert calls == ["_scan_sync", "_scan_sync"]

    async def test_presidio_always_offloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("presidio_analyzer")
        try:
            pii = PII(engine="presidio")
        except Exception as exc:  # model not installed
            pytest.skip(f"presidio unavailable: {type(exc).__name__}")
        calls = self._spy(monkeypatch)
        await pii.scan("John Smith lives in Boston.")
        assert calls == ["_scan_sync"]
