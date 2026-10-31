"""The MySQL/MariaDB test servers: compose file, make targets, CI jobs and extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parents[2]
COMPOSE = PACKAGE / "docker" / "test-dbs.compose.yml"
CI = PACKAGE.parents[1] / ".github" / "workflows" / "ci.yml"


def test_compose_defines_exactly_the_two_test_servers() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    expected = {
        "mysql": ("mysql:8.4@sha256:", "dendrux-test-mysql", "53306", "utf8mb4_0900_bin"),
        "mariadb": ("mariadb:10.11@sha256:", "dendrux-test-mariadb", "53307", "utf8mb4_nopad_bin"),
    }
    assert set(services) == set(expected)
    for name, (image, container, port, collation) in expected.items():
        service = services[name]
        assert service["image"].startswith(image)
        assert service["container_name"] == container
        assert service["ports"] == [f"127.0.0.1:{port}:3306"]
        assert service["mem_limit"] == "1g"
        assert f"--collation-server={collation}" in service["command"]
        assert "--character-set-server=utf8mb4" in service["command"]
        assert "utf8mb4_bin" not in " ".join(service["command"])
        assert "volumes" not in service
        assert "healthcheck" in service


def test_make_targets_start_and_remove_only_the_compose_project() -> None:
    makefile = (PACKAGE / "Makefile").read_text()
    assert "TEST_DBS = docker compose -f docker/test-dbs.compose.yml" in makefile
    assert "test-dbs-up:\n\t$(TEST_DBS) up -d --wait" in makefile
    assert "test-dbs-down:\n\t$(TEST_DBS) down -v" in makefile
    assert "test-mysql:" in makefile
    phony = next(line for line in makefile.splitlines() if line.startswith(".PHONY"))
    assert {"test-dbs-up", "test-dbs-down", "test-mysql"} <= set(phony.split())


def test_ci_runs_both_servers_from_the_compose_file_and_fails_on_skips() -> None:
    job = yaml.safe_load(CI.read_text())["jobs"]["mysql-family"]
    servers = {entry["server"]: entry for entry in job["strategy"]["matrix"]["include"]}
    assert servers["mysql"]["env_name"] == "DENDRUX_TEST_MYSQL_URL"
    assert servers["mariadb"]["env_name"] == "DENDRUX_TEST_MARIADB_URL"
    steps = {step.get("name", ""): step for step in job["steps"]}
    start = steps["Start ${{ matrix.server }}"]["run"]
    assert (
        start == "docker compose -f docker/test-dbs.compose.yml up -d --wait ${{ matrix.server }}"
    )
    test = steps["Test on ${{ matrix.server }}"]
    assert "_test" in test["env"]["SERVER_URL"]
    assert "pytest tests/db" in test["run"]
    assert "not set' pytest.log" in test["run"] and "exit 1" in test["run"]
    remove = steps["Remove ${{ matrix.server }}"]
    assert remove["if"] == "always()"
    assert "down -v" in remove["run"]
    assert '".[dev,mysql]"' in steps["Install dependencies"]["run"]


def test_extras_publish_the_driver_and_the_test_kit_dependencies() -> None:
    extras = tomllib.loads((PACKAGE / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    assert extras["mysql"] == ["asyncmy>=0.2.10"]
    names = sorted(requirement.split(">")[0].split("=")[0] for requirement in extras["testkit"])
    assert names == ["pytest", "pytest-asyncio"]
