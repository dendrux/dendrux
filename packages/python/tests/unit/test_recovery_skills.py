"""Versioned skill activation snapshots do not persist configuration or payloads."""

from dendrux import Agent, Skill
from dendrux.llm.mock import MockLLM
from dendrux.skills._session import SkillSession


def test_skills_snapshot_roundtrip_and_unknown_name_count(caplog):
    agent = Agent(
        prompt="p",
        provider=MockLLM([]),
        skills=[
            Skill(name="a", description="d", body="b"),
            Skill(name="b", description="d", body="b"),
        ],
    )
    session = SkillSession.prepare(agent, explicit=["b", "a"])
    assert hasattr(session, "snapshot"), "skills need their versioned activation serializer"
    assert session.snapshot() == {"v": 1, "activated": ["a", "b"]}
    restored = SkillSession.prepare(agent)
    restored.run_id = "00000000000000000000000401"
    restored.restore_snapshot({"v": 1, "activated": ["a", "unknown"]})
    assert restored.snapshot() == {"v": 1, "activated": ["a"]}
    assert any(
        getattr(r, "code", None) == "skills_session_unknown"
        and r.run_id == "00000000000000000000000401"
        and r.count == 1
        for r in caplog.records
    )
