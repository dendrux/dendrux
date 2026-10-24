"""Guards for the repository's CI and supply-chain configuration.

These read ``.github/`` and ``pyproject.toml`` directly, so a later edit that
unpins an action, widens a permission or drops the Postgres requirement fails
here instead of going unnoticed.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[4]
GITHUB = REPO / ".github"
PYPROJECT = REPO / "packages" / "python" / "pyproject.toml"

pytestmark = pytest.mark.skipif(not GITHUB.is_dir(), reason="repository checkout only")

WORKFLOWS = sorted((GITHUB / "workflows").glob("*.yml")) if GITHUB.is_dir() else []
SHA = re.compile(r"^[0-9a-f]{40}$")


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, GITHUB / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(name: str) -> dict:
    return yaml.safe_load((GITHUB / "workflows" / name).read_text())


def _all_steps(workflow: dict) -> list[dict]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


class TestWorkflows:
    def test_there_are_workflows(self) -> None:
        assert {w.name for w in WORKFLOWS} >= {
            "ci.yml",
            "codeql.yml",
            "release.yml",
            "security.yml",
        }

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_every_action_is_pinned_to_a_commit_sha(self, path: Path) -> None:
        for line in path.read_text().splitlines():
            match = re.match(r"\s*-?\s*uses:\s*(\S+)", line)
            if not match or match.group(1).startswith("./"):
                continue
            _, _, ref = match.group(1).partition("@")
            assert SHA.match(ref), f"{path.name}: {match.group(1)} is not pinned to a SHA"
            assert re.search(r"#\s*v\d", line), f"{path.name}: add the version comment: {line}"

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_top_level_permissions_are_read_only(self, path: Path) -> None:
        permissions = yaml.safe_load(path.read_text()).get("permissions")
        assert permissions == {"contents": "read"}, f"{path.name} needs `contents: read` only"

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_jobs_have_timeouts_and_no_dangerous_triggers(self, path: Path) -> None:
        workflow = yaml.safe_load(path.read_text())
        triggers = workflow.get("on") or workflow.get(True)
        assert "pull_request_target" not in triggers
        for name, job in workflow["jobs"].items():
            assert "timeout-minutes" in job, f"{path.name}:{name} has no timeout"

    @pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
    def test_checkout_does_not_persist_credentials(self, path: Path) -> None:
        for step in _all_steps(yaml.safe_load(path.read_text())):
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") is False

    def test_no_secrets_are_referenced(self) -> None:
        for path in WORKFLOWS:
            assert "secrets." not in path.read_text(), path.name

    def test_job_level_write_permissions_are_narrow(self) -> None:
        allowed = {
            "codeql.yml": {"security-events"},
            "release.yml": {"id-token"},
        }
        for path in WORKFLOWS:
            for job in yaml.safe_load(path.read_text())["jobs"].values():
                writes = {k for k, v in job.get("permissions", {}).items() if v == "write"}
                assert writes <= allowed.get(path.name, set()), (path.name, writes)


class TestPostgresRequirement:
    def test_ci_fails_when_the_postgres_url_is_missing_or_tests_skip(self) -> None:
        steps = {s.get("name"): s for s in _all_steps(_workflow("ci.yml"))}
        guard = steps["Require the Postgres backend"]["run"]
        assert "DENDRUX_TEST_PG_URL" in guard
        assert "exit 1" in guard
        test = steps["Test"]
        assert test["shell"] == "bash"  # pipefail, so pytest's own failure still counts
        assert "-rs" in test["run"]
        assert "DENDRUX_TEST_PG_URL not set" in test["run"]
        assert "exit 1" in test["run"]

    def test_postgres_service_and_env_are_defined(self) -> None:
        job = _workflow("ci.yml")["jobs"]["python"]
        assert "postgres" in job["services"]
        assert job["env"]["DENDRUX_TEST_PG_URL"].startswith("postgresql+asyncpg://")


class TestRelease:
    def test_publish_uses_trusted_publishing_behind_a_reviewed_environment(self) -> None:
        release = _workflow("release.yml")
        publish = release["jobs"]["publish"]
        assert publish["environment"]["name"] == "pypi"
        assert publish["permissions"] == {"id-token": "write"}
        assert "PYPI_TRUSTED_PUBLISHING" in publish["if"]
        assert publish["needs"] == "build"
        uses = [s.get("uses", "") for s in publish["steps"]]
        assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in uses)
        assert "password" not in str(publish)

    def test_build_checks_the_tag_and_the_distributions(self) -> None:
        text = "\n".join(s.get("run", "") for s in _all_steps(_workflow("release.yml")))
        assert "merge-base --is-ancestor" in text
        assert "twine check --strict" in text
        assert "check_wheel_dashboard.py" in text

    def test_only_version_tags_trigger_it(self) -> None:
        release = _workflow("release.yml")
        trigger = release.get("on") or release[True]
        assert trigger == {"push": {"tags": ["v*"]}}


class TestDependabot:
    def test_covers_pip_and_actions(self) -> None:
        config = yaml.safe_load((GITHUB / "dependabot.yml").read_text())
        ecosystems = {u["package-ecosystem"]: u for u in config["updates"]}
        assert set(ecosystems) == {"pip", "github-actions"}
        assert ecosystems["pip"]["directory"] == "/packages/python"
        assert ecosystems["github-actions"]["directory"] == "/"


class TestPackaging:
    @pytest.fixture(scope="class")
    def project(self) -> dict:
        return tomllib.loads(PYPROJECT.read_text())

    def test_uvicorn_is_bounded(self, project: dict) -> None:
        for extra in ("http",):
            uvicorn = next(
                d
                for d in project["project"]["optional-dependencies"][extra]
                if d.startswith("uvicorn")
            )
            assert "<" in uvicorn and ">=" in uvicorn

    def test_dev_tools_are_bounded_or_pinned(self, project: dict) -> None:
        for dep in project["project"]["optional-dependencies"]["dev"]:
            if dep.startswith("dendrux["):
                continue
            assert "==" in dep or "<" in dep, f"unbounded dev dependency: {dep}"

    def test_ruff_matches_pre_commit(self, project: dict) -> None:
        pinned = next(
            d for d in project["project"]["optional-dependencies"]["dev"] if d.startswith("ruff")
        ).split("==")[1]
        pre_commit = yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text())
        rev = next(r["rev"] for r in pre_commit["repos"] if "ruff" in r["repo"])
        assert rev == f"v{pinned}"

    def test_cli_is_inside_the_coverage_gate(self, project: dict) -> None:
        omit = project["tool"]["coverage"]["run"]["omit"]
        assert not any("cli" in pattern for pattern in omit)

    def test_gitignore_points_at_the_real_package_path(self) -> None:
        text = (REPO / ".gitignore").read_text()
        assert "src/dendrite/" not in text
        assert "packages/python/src/dendrux/dashboard/static/*" in text


class TestAuditGate:
    gate = _load_script("audit_gate") if GITHUB.is_dir() else None
    TODAY = dt.date(2026, 10, 24)

    @staticmethod
    def _audit(*vulns: dict) -> dict:
        return {"dependencies": [{"name": "pkg", "version": "1.0", "vulns": list(vulns)}]}

    def test_clean_audit_passes(self) -> None:
        assert self.gate.evaluate({"dependencies": []}, {"entries": []}, self.TODAY) == []

    def test_unlisted_vulnerability_fails(self) -> None:
        audit = self._audit({"id": "PYSEC-1", "aliases": [], "fix_versions": ["1.1"]})
        problems = self.gate.evaluate(audit, {"entries": []}, self.TODAY)
        assert len(problems) == 1
        assert "PYSEC-1" in problems[0]

    def test_allowlisted_by_id_or_alias_passes(self) -> None:
        entry = {"id": "GHSA-x", "reason": "unreachable", "expires": "2026-11-30"}
        by_alias = self._audit({"id": "PYSEC-1", "aliases": ["GHSA-x"], "fix_versions": []})
        assert self.gate.evaluate(by_alias, {"entries": [entry]}, self.TODAY) == []
        by_id = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        assert self.gate.evaluate(by_id, {"entries": [entry]}, self.TODAY) == []

    def test_expired_entry_fails_even_when_unmatched(self) -> None:
        entry = {"id": "GHSA-x", "reason": "r", "expires": "2026-10-23"}
        problems = self.gate.evaluate({"dependencies": []}, {"entries": [entry]}, self.TODAY)
        assert any("expired" in p for p in problems)

    def test_expired_entry_no_longer_suppresses_its_vulnerability(self) -> None:
        entry = {"id": "GHSA-x", "reason": "r", "expires": "2026-10-23"}
        audit = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        problems = self.gate.evaluate(audit, {"entries": [entry]}, self.TODAY)
        assert len(problems) == 2

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": "GHSA-x", "expires": "2026-11-01"},
            {"id": "GHSA-x", "reason": " ", "expires": "2026-11-01"},
            {"reason": "r", "expires": "2026-11-01"},
            {"id": "GHSA-x", "reason": "r"},
            {"id": "GHSA-x", "reason": "r", "expires": "soon"},
            {"id": "GHSA-x", "reason": "r", "expires": "2027-12-31"},
        ],
    )
    def test_malformed_or_distant_entries_are_rejected(self, entry: dict) -> None:
        audit = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        problems = self.gate.evaluate(audit, {"entries": [entry]}, self.TODAY)
        assert len(problems) == 2  # the entry itself and the vulnerability it fails to cover

    def test_shipped_allowlist_is_valid_today(self) -> None:
        import json

        data = json.loads((GITHUB / "pip-audit-allowlist.json").read_text())
        _, problems = self.gate.check_allowlist(data, dt.datetime.now(dt.UTC).date())
        assert problems == []

    def test_unreadable_report_fails_the_build(self, tmp_path: Path) -> None:
        allow = tmp_path / "allow.json"
        allow.write_text('{"entries": []}')
        assert self.gate.main(["gate", str(tmp_path / "missing.json"), str(allow)]) == 1
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert self.gate.main(["gate", str(bad), str(allow)]) == 1

    def test_main_exit_codes(self, tmp_path: Path) -> None:
        import json

        allow = tmp_path / "allow.json"
        allow.write_text('{"entries": []}')
        clean = tmp_path / "clean.json"
        clean.write_text(json.dumps({"dependencies": []}))
        dirty = tmp_path / "dirty.json"
        dirty.write_text(json.dumps(self._audit({"id": "X-1", "aliases": []})))
        assert self.gate.main(["gate", str(clean), str(allow)]) == 0
        assert self.gate.main(["gate", str(dirty), str(allow)]) == 1
        assert self.gate.main(["gate"]) == 2


class TestWheelDashboardCheck:
    check = _load_script("check_wheel_dashboard") if GITHUB.is_dir() else None
    S = "dendrux/dashboard/static/"

    def test_consistent_build_passes(self) -> None:
        html = '<script src="./assets/index-abc.js"></script><link href="./assets/i.css">'
        names = {f"{self.S}index.html", f"{self.S}assets/index-abc.js", f"{self.S}assets/i.css"}
        assert self.check.problems(names, html) == []

    def test_reference_to_a_missing_asset_fails(self) -> None:
        html = '<script src="./assets/index-new.js"></script>'
        names = {f"{self.S}index.html", f"{self.S}assets/index-old.js"}
        problems = self.check.problems(names, html)
        assert len(problems) == 1
        assert "index-new.js" in problems[0]

    def test_missing_index_fails(self) -> None:
        assert self.check.problems({"dendrux/__init__.py"}, None)

    def test_index_without_assets_fails(self) -> None:
        names = {f"{self.S}index.html"}
        assert self.check.problems(names, "<html></html>")

    def test_committed_build_is_consistent(self) -> None:
        static = REPO / "packages" / "python" / "src" / "dendrux" / "dashboard" / "static"
        names = {
            f"{self.S}{p.relative_to(static).as_posix()}" for p in static.rglob("*") if p.is_file()
        }
        html = (static / "index.html").read_text()
        assert self.check.problems(names, html) == []

    def test_main_reads_a_wheel(self, tmp_path: Path) -> None:
        import zipfile

        wheel = tmp_path / "w.whl"
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr(f"{self.S}index.html", '<script src="./assets/a.js"></script>')
            z.writestr(f"{self.S}assets/a.js", "1")
        assert self.check.main(["check", str(wheel)]) == 0
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr(f"{self.S}index.html", '<script src="./assets/a.js"></script>')
        assert self.check.main(["check", str(wheel)]) == 1
        empty = tmp_path / "e.whl"
        with zipfile.ZipFile(empty, "w") as z:
            z.writestr("x", "1")
        assert self.check.main(["check", str(empty)]) == 1
        assert self.check.main(["check"]) == 2
