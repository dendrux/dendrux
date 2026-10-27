"""The upgrade procedure selects the new migration distribution before rolling workers."""

from pathlib import Path

PAGE = Path(__file__).resolve().parents[4] / "docs/upgrading-to-0.5.0.mdx"


def test_migration_job_installs_distribution_before_migrating():
    text = PAGE.read_text()
    steps = text.split("## The upgrade in four steps", 1)[1].split("### Mixed fleets", 1)[0]
    assert (
        steps.index("pip install")
        < steps.index("dendrux db migrate")
        < steps.index("Roll your workers")
    )
    assert "create the new indexes by hand, or" not in text
    assert "stamp `c6d7e8f9a0b1`" in text
    assert "without persistence" in text


def test_mixed_fleet_explains_idempotency_restriction():
    text = PAGE.read_text().split("### Mixed fleets", 1)[1].split("## ", 1)[0]
    assert "idempotency" in text.lower()
    assert "0.4.1" in text
    assert "across tenants" in text
