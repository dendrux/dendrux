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


class TestCodeQL:
    def test_covers_python_actions_and_the_dashboard_frontend(self) -> None:
        job = _workflow("codeql.yml")["jobs"]["analyze"]
        assert set(job["strategy"]["matrix"]["language"]) == {
            "python",
            "actions",
            "javascript-typescript",
        }


class TestDashboardFrontendJob:
    def test_frontend_is_linted_tested_and_its_committed_build_verified(self) -> None:
        job = _workflow("ci.yml")["jobs"]["dashboard"]
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        for command in ("npm ci", "npm run lint", "npm test", "npm run build"):
            assert command in runs
        assert "git diff --exit-code" in runs
        assert "git ls-files" in runs

    def test_frontend_test_suite_exists(self) -> None:
        tests = list((REPO / "packages" / "dashboard" / "src").rglob("*.test.tsx"))
        assert tests


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

    def test_publish_job_is_limited_to_this_repository_and_tags(self) -> None:
        condition = _workflow("release.yml")["jobs"]["publish"]["if"]
        for fragment in (
            "vars.PYPI_TRUSTED_PUBLISHING == 'true'",
            "github.repository == 'dendrux/dendrux'",
            "github.ref_type == 'tag'",
            "startsWith(github.ref_name, 'v')",
        ):
            assert fragment in condition

    def test_publish_rechecks_that_distributions_match_the_tag(self) -> None:
        steps = _workflow("release.yml")["jobs"]["publish"]["steps"]
        run = next(s["run"] for s in steps if s.get("name") == "Distributions must match the tag")
        assert "GITHUB_REF_NAME" in run
        assert "exit 1" in run

    def test_owner_checklist_is_ordered_and_names_every_boundary(self) -> None:
        text = (GITHUB / "RELEASING.md").read_text()
        steps = re.split(r"\n(?=\d\. \*\*)", text.split("## Releasing")[0])[1:]
        assert len(steps) == 6
        expected = [
            ("`pypi` environment", "required reviewer", "`v*`"),
            ("tag ruleset", "`v*`"),
            ("Protect `main`",),
            ("PyPI trusted publisher", "environment **`pypi`**"),
            ("organisation-level variable",),
            ("Only now", "PYPI_TRUSTED_PUBLISHING"),
        ]
        for step, fragments in zip(steps, expected, strict=True):
            for fragment in fragments:
                assert fragment in step, (fragment, step)

    def test_security_policy_does_not_claim_publishing_is_live(self) -> None:
        text = (REPO / "SECURITY.md").read_text()
        assert "is being enabled" in text
        assert ".github/RELEASING.md" in text

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
        assert set(ecosystems) == {"pip", "npm", "github-actions"}
        assert ecosystems["npm"]["directory"] == "/packages/dashboard"
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

    def test_dashboard_build_is_a_hatch_artifact(self, project: dict) -> None:
        targets = project["tool"]["hatch"]["build"]["targets"]
        for target in ("wheel", "sdist"):
            assert "src/dendrux/dashboard/static/**" in targets[target]["artifacts"]

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
    NONE: frozenset[str] = frozenset()

    @staticmethod
    def _audit(*vulns: dict, name: str = "pkg", version: str = "1.0") -> dict:
        return {"dependencies": [{"name": name, "version": version, "vulns": list(vulns)}]}

    @staticmethod
    def _inventory_of(audit: dict) -> list[dict]:
        return [
            {"name": d["name"], "version": d.get("version", "1.0")}
            for d in audit.get("dependencies", [])
            if isinstance(d, dict) and "name" in d
        ]

    def _ev(
        self,
        audit: dict,
        allow: dict | None = None,
        *,
        inventory: object = "auto",
        status: int = 0,
        expected: frozenset[str] | None = None,
    ) -> list[str]:
        inv = self._inventory_of(audit) if inventory == "auto" else inventory
        return self.gate.evaluate(
            audit,
            allow or {"entries": []},
            self.TODAY,
            inv,
            status,
            self.NONE if expected is None else expected,
        )

    def test_clean_audit_passes(self) -> None:
        assert self._ev(self._audit()) == []

    def test_unlisted_vulnerability_fails(self) -> None:
        audit = self._audit({"id": "PYSEC-1", "aliases": [], "fix_versions": ["1.1"]})
        problems = self._ev(audit, status=1)
        assert len(problems) == 1
        assert "PYSEC-1" in problems[0]

    def test_allowlisted_by_id_or_alias_passes(self) -> None:
        entry = {"id": "GHSA-x", "reason": "unreachable", "expires": "2026-11-30"}
        by_alias = self._audit({"id": "PYSEC-1", "aliases": ["GHSA-x"], "fix_versions": []})
        assert self._ev(by_alias, {"entries": [entry]}, status=1) == []
        by_id = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        assert self._ev(by_id, {"entries": [entry]}, status=1) == []

    def test_expired_entry_fails_even_when_unmatched(self) -> None:
        entry = {"id": "GHSA-x", "reason": "r", "expires": "2026-10-23"}
        assert any("expired" in p for p in self._ev(self._audit(), {"entries": [entry]}))

    def test_expired_entry_no_longer_suppresses_its_vulnerability(self) -> None:
        entry = {"id": "GHSA-x", "reason": "r", "expires": "2026-10-23"}
        audit = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        assert len(self._ev(audit, {"entries": [entry]}, status=1)) == 2

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
        assert len(self._ev(audit, {"entries": [entry]}, status=1)) == 2

    @pytest.mark.parametrize(
        "report",
        [
            {},
            {"dependencies": []},
            {"dependencies": "nope"},
            {"dependencies": [{"name": "pkg", "version": "1"}]},  # no vulns field
            {"dependencies": [{"name": "pkg", "vulns": "x"}]},
            {"dependencies": [{"version": "1", "vulns": []}]},
            {"dependencies": ["pkg"]},
        ],
    )
    def test_malformed_reports_fail(self, report: dict) -> None:
        assert self._ev(report, inventory=[{"name": "pkg", "version": "1"}])

    def test_unaudited_dependency_fails(self) -> None:
        report = {"dependencies": [{"name": "pkg", "skip_reason": "could not be audited"}]}
        problems = self._ev(report, inventory=[{"name": "pkg", "version": "1"}])
        assert len(problems) == 1
        assert "not audited" in problems[0]

    def test_package_under_test_may_be_skipped_or_absent(self) -> None:
        inventory = [
            {"name": "dendrux", "version": "0.5.0.dev1"},
            {"name": "pkg", "version": "1.0"},
        ]
        skipped = {
            "dependencies": [
                {"name": "dendrux", "skip_reason": "Dependency not found on PyPI"},
                {"name": "pkg", "version": "1.0", "vulns": []},
            ]
        }
        assert self._ev(skipped, inventory=inventory) == []
        absent = self._audit()
        assert self._ev(absent, inventory=inventory) == []

    def test_installed_package_missing_from_the_report_fails(self) -> None:
        inventory = [{"name": "pkg", "version": "1.0"}, {"name": "forgotten", "version": "2.3"}]
        problems = self._ev(self._audit(), inventory=inventory)
        assert problems == ["forgotten 2.3: installed but missing from the audit report."]

    def test_report_record_without_a_version_fails(self) -> None:
        report = {"dependencies": [{"name": "pkg", "vulns": []}]}
        problems = self._ev(report, inventory=[{"name": "pkg", "version": "1.0"}])
        assert problems == ["pkg: audit record has no version."]

    def test_version_mismatch_fails(self) -> None:
        problems = self._ev(
            self._audit(version="1.0"), inventory=[{"name": "pkg", "version": "2.0"}]
        )
        assert problems == ["pkg: audited 1.0 but 2.0 is installed."]

    def test_names_match_by_pep_503_normalisation(self) -> None:
        report = {"dependencies": [{"name": "Typing_Extensions", "version": "4", "vulns": []}]}
        assert self._ev(report, inventory=[{"name": "typing-extensions", "version": "4"}]) == []

    @pytest.mark.parametrize(
        "inventory", [None, [], "x", [{"name": "pkg"}], [{"version": "1"}], [1]]
    )
    def test_missing_or_malformed_inventory_fails(self, inventory: object) -> None:
        problems = self._ev(self._audit(), inventory=inventory)
        assert any("inventory" in p for p in problems)

    def test_expected_packages_must_be_installed_and_audited(self) -> None:
        problems = self._ev(self._audit(name="other"), expected=frozenset({"pydantic"}))
        assert any("pydantic" in p and "not installed" in p for p in problems)
        report = {
            "dependencies": [
                {"name": "Pydantic", "version": "2", "vulns": []},
                {"name": "httpx", "version": "1", "vulns": []},
            ]
        }
        assert self._ev(report, expected=frozenset({"pydantic", "httpx"})) == []

    def test_a_skipped_expected_package_still_fails(self) -> None:
        report = {"dependencies": [{"name": "pydantic", "skip_reason": "x"}]}
        problems = self._ev(report, expected=frozenset({"pydantic"}))
        assert any("not audited" in p for p in problems)

    @pytest.mark.parametrize("status", [2, 3, 127, 130, -9])
    def test_unexpected_auditor_status_fails(self, status: int) -> None:
        problems = self._ev(self._audit(), status=status)
        assert any("exited with status" in p for p in problems)

    def test_status_one_without_findings_is_an_auditor_error(self) -> None:
        problems = self._ev(self._audit(), status=1)
        assert any("no findings" in p for p in problems)

    def test_status_one_with_allowlisted_findings_passes(self) -> None:
        entry = {"id": "GHSA-x", "reason": "r", "expires": "2026-11-30"}
        audit = self._audit({"id": "GHSA-x", "aliases": [], "fix_versions": []})
        assert self._ev(audit, {"entries": [entry]}, status=1) == []

    def test_default_expected_set_covers_core_packages(self) -> None:
        assert {"pydantic", "sqlalchemy", "httpx", "fastapi"} <= self.gate.EXPECTED_AUDITED

    def test_shipped_allowlist_is_valid_today(self) -> None:
        import json

        data = json.loads((GITHUB / "pip-audit-allowlist.json").read_text())
        _, problems = self.gate.check_allowlist(data, dt.datetime.now(dt.UTC).date())
        assert problems == []

    def test_main_exit_codes(self, tmp_path: Path) -> None:
        import json

        names = sorted(self.gate.EXPECTED_AUDITED)
        inventory = tmp_path / "inventory.json"
        inventory.write_text(json.dumps([{"name": n, "version": "1"} for n in names]))
        allow = tmp_path / "allow.json"
        allow.write_text('{"entries": []}')
        records = [{"name": n, "version": "1", "vulns": []} for n in names]
        clean = tmp_path / "clean.json"
        clean.write_text(json.dumps({"dependencies": records}))
        dirty = tmp_path / "dirty.json"
        dirty.write_text(
            json.dumps(
                {
                    "dependencies": [
                        {**records[0], "vulns": [{"id": "X-1", "aliases": []}]},
                        *records[1:],
                    ]
                }
            )
        )
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"dependencies": records[:1]}))
        empty = tmp_path / "empty.json"
        empty.write_text("{}")

        def run(report: Path, status: str = "0", inv: Path = inventory) -> int:
            return self.gate.main(["gate", str(report), str(allow), str(inv), status])

        assert run(clean) == 0
        assert run(dirty, "1") == 1
        assert run(empty) == 1
        assert run(partial) == 1  # installed packages missing from the report
        assert run(clean, "2") == 1
        assert run(clean, "not-a-number") == 1
        assert run(tmp_path / "missing.json") == 1
        assert run(clean, inv=tmp_path / "missing-inventory.json") == 1
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert run(bad) == 1
        assert self.gate.main(["gate", str(clean)]) == 2


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

    def test_committed_file_missing_from_the_wheel_fails(self, tmp_path: Path) -> None:
        static = tmp_path / "static"
        (static / "assets").mkdir(parents=True)
        (static / "index.html").write_text("x")
        (static / "assets" / "lazy-chunk.js").write_text("x")
        names = {f"{self.S}index.html"}
        problems = self.check.missing_from_wheel(names, static)
        assert problems == [f"{self.S}assets/lazy-chunk.js is committed but not in the wheel."]
        assert (
            self.check.missing_from_wheel(names | {f"{self.S}assets/lazy-chunk.js"}, static) == []
        )

    def test_real_static_dir_is_fully_in_a_wheel_listing(self) -> None:
        static = REPO / "packages" / "python" / "src" / "dendrux" / "dashboard" / "static"
        names = {
            f"{self.S}{p.relative_to(static).as_posix()}" for p in static.rglob("*") if p.is_file()
        }
        assert self.check.missing_from_wheel(names, static) == []

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
