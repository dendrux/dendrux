"""Tests for database enums."""

from __future__ import annotations

from dendrux.db.enums import AgentRunStatus


class TestAgentRunStatus:
    def test_has_all_values(self) -> None:
        values = {s.value for s in AgentRunStatus}
        assert values == {
            "pending",
            "running",
            "success",
            "error",
            "max_iterations",
            "cancelled",
            "waiting_client_tool",
            "waiting_human_input",
            "waiting_approval",
        }

    def test_is_str_enum(self) -> None:
        """Status values can be used directly as strings."""
        assert AgentRunStatus.RUNNING == "running"
        assert f"status={AgentRunStatus.SUCCESS}" == "status=success"

    def test_value_matches_name_lowercase(self) -> None:
        for status in AgentRunStatus:
            assert status.value == status.name.lower()
