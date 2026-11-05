"""Both storage formats must run on serial Postgres CI, with dialect pause coverage."""

from tests.conformance.test_kit_live_ci import jobs, step


def test_storage_format_goldens_run_serially_on_postgres():
    check = step("python", "Storage-format conformance on SQLite and Postgres")
    assert "matrix.shard == 1" in check["if"]
    env = check["env"]
    database = env["DENDRUX_TEST_PG_URL"].rsplit("/", 1)[1]
    assert database.endswith("_test") and database != "dendrux_test"
    assert env["DENDRUX_CONFORMANCE_PG_DB"] == database
    command = check["run"]
    assert "tests/conformance/test_conformance.py" in command
    assert "tests/conformance/test_format2.py" in command
    assert "-n 0" in command and "--no-cov" in command
    assert "skipped" in command and "exit 1" in command


def test_mysql_family_runs_clarification_projection_contract():
    check = step("mysql-family", "Pause-summary contract on")
    assert "tests/conformance/test_pause_summary_formats.py" in check["run"]
    assert "matrix.server" in check["run"]
    assert "-n 0" in check["run"]
    assert "skipped" in check["run"] and "exit 1" in check["run"]
    assert jobs()["mysql-family"]["strategy"]["matrix"]["include"]
