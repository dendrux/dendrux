"""Response limits and wire outcomes are documented where applications configure them."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def test_provider_error_and_bounds_have_reference_entries():
    errors = (REPO / "docs/reference/errors.mdx").read_text()
    recipe = (REPO / "docs/recipes/handling-provider-errors.mdx").read_text()
    models = (REPO / "docs/architecture/models.mdx").read_text()
    assert "| `ProviderResponseError` | `rejected` |" in errors
    assert "| `ProviderResponseError` | `rejected` |" in recipe
    assert "| `ProviderResponseError` | 502 |" in recipe
    for text in (recipe, models):
        assert "| `max_stream_bytes` |" in text
        assert "| `max_stream_tool_calls` |" in text


def test_wire_and_pending_explanations_precede_related_links():
    agent = (REPO / "docs/reference/agent.mdx").read_text()
    store = (REPO / "docs/reference/run-store.mdx").read_text()
    assert agent.index("### Wire answer display bound") < agent.index("## Where this fits")
    assert agent.index("| `wire_version` |") < agent.index("| `answer_omitted` |")
    assert store.index("A requested pause that retains server-target") < store.index(
        "## Where this fits"
    )


def test_unstamped_upgrade_has_an_executable_stamp_command():
    text = (REPO / "docs/upgrading-to-0.5.0.mdx").read_text()
    steps = text.split("## The upgrade in four steps", 1)[1].split("### Mixed fleets", 1)[0]
    assert "alembic stamp c6d7e8f9a0b1" in steps
    assert "script_location" in steps
    assert "\n In a rehearsal" not in text
