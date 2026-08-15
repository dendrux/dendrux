"""Build and verify Dendrux MCP support from isolated distribution artifacts."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
ECHO_SERVER = PACKAGE_ROOT / "tests" / "fixtures" / "mcp_echo_server.py"


def _run(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _only_artifact(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"Expected one {pattern} artifact, found {matches!r}.")
    return matches[0]


def _assert_artifact_contents(wheel: Path, sdist: Path) -> None:
    required_wheel_suffixes = {
        "dendrux/mcp/__init__.py",
        "dendrux/mcp/_client.py",
        "dendrux/mcp/_errors.py",
        "dendrux/mcp/_observability.py",
        "dendrux/mcp/_runtime.py",
        "dendrux/mcp/_server.py",
        "dendrux/mcp/_source.py",
        "dendrux/py.typed",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = required_wheel_suffixes - names
        if missing:
            raise AssertionError(f"Wheel is missing required files: {sorted(missing)!r}.")
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata_text = archive.read(metadata_name).decode()

    if "Provides-Extra: mcp" not in metadata_text:
        raise AssertionError("Wheel metadata does not publish the 'mcp' extra.")
    mcp_requirements = [
        line for line in metadata_text.splitlines() if line.lower().startswith("requires-dist: mcp")
    ]
    if not mcp_requirements or not any("extra == 'mcp'" in line for line in mcp_requirements):
        raise AssertionError("Wheel metadata does not attach the MCP SDK to the 'mcp' extra.")

    required_sdist_suffixes = {
        "/README.md",
        "/examples/31_mcp_runtime_multi_user.py",
        "/examples/32_mcp_github_remote.py",
        "/src/dendrux/mcp/_runtime.py",
    }
    with tarfile.open(sdist, "r:gz") as archive:
        names = set(archive.getnames())
    missing = {
        suffix
        for suffix in required_sdist_suffixes
        if not any(name.endswith(suffix) for name in names)
    }
    if missing:
        raise AssertionError(f"Source distribution is missing required files: {sorted(missing)!r}.")


def _venv_python(directory: Path) -> Path:
    executable = "python.exe" if os.name == "nt" else "python"
    scripts = "Scripts" if os.name == "nt" else "bin"
    return directory / scripts / executable


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


def _verify_installed_artifact(
    *, artifact: Path, expected_version: str, fixture: Path, work_directory: Path
) -> None:
    environment_directory = work_directory / f"venv-{artifact.name}"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment_directory)
    python = _venv_python(environment_directory)
    environment = _clean_environment()
    requirement = f"dendrux[mcp] @ {artifact.resolve().as_uri()}"

    _run(
        [
            python.as_posix(),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            requirement,
        ],
        cwd=work_directory,
        environment=environment,
    )
    _run(
        [
            python.as_posix(),
            "-I",
            Path(__file__).resolve().as_posix(),
            "--installed-check",
            "--expected-version",
            expected_version,
            "--fixture",
            fixture.resolve().as_posix(),
        ],
        cwd=work_directory,
        environment=environment,
    )


async def _exercise_installed_runtime(fixture: Path) -> None:
    from dendrux import Agent
    from dendrux.mcp import (
        MCPConnectionClosed,
        MCPRuntime,
        MCPRuntimeEvent,
        MCPSource,
    )

    events: list[MCPRuntimeEvent] = []

    class Observer:
        def on_event(self, event: MCPRuntimeEvent) -> None:
            events.append(event)

    runtime = MCPRuntime(observer=Observer(), idle_timeout=None, shutdown_timeout=2.0)
    connection = runtime.bind(
        tenant_key="distribution-smoke",
        connection_key="echo",
        source=MCPSource.stdio(
            "echo",
            [sys.executable, fixture.as_posix()],
            connect_timeout=10.0,
            call_timeout=10.0,
        ),
    )
    agent = Agent(prompt="distribution smoke", tool_sources=[connection.tools()])

    try:
        lookups = await agent.get_tool_lookups()
        result = await lookups.fn["echo__echo"](message="installed artifact")
        if result != {"result": "installed artifact"}:
            raise AssertionError(f"Unexpected MCP tool result: {result!r}.")
    finally:
        await agent.close()
        await runtime.close()

    closed = [event for event in events if isinstance(event, MCPConnectionClosed)]
    if len(closed) != 1 or not closed[0].clean:
        raise AssertionError(f"Expected one clean physical connection close, got {closed!r}.")


def _installed_check(*, expected_version: str, fixture: Path) -> None:
    import mcp

    import dendrux
    from dendrux.mcp import MCPRuntime, MCPRuntimeSnapshot, MCPToolView

    installed_version = importlib.metadata.version("dendrux")
    if installed_version != expected_version:
        raise AssertionError(
            f"Installed Dendrux version {installed_version!r} != {expected_version!r}."
        )
    installed_root = Path(dendrux.__file__).resolve()
    if not installed_root.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(
            f"Imported Dendrux outside the isolated environment: {installed_root}."
        )
    if importlib.metadata.version("mcp").split(".", maxsplit=1)[0] != "2":
        raise AssertionError("The dendrux[mcp] extra did not install MCP SDK v2.")
    if not all((MCPRuntime, MCPRuntimeSnapshot, MCPToolView, mcp)):
        raise AssertionError("Public MCP imports did not resolve.")

    asyncio.run(_exercise_installed_runtime(fixture))
    print(f"PASS: dendrux {installed_version} from {installed_root}")


def _build_and_verify() -> None:
    project = tomllib.loads(PYPROJECT.read_text())
    expected_version = project["project"]["version"]

    with tempfile.TemporaryDirectory(prefix="dendrux-distribution-") as temporary:
        work_directory = Path(temporary)
        artifact_directory = work_directory / "artifacts"
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                artifact_directory.as_posix(),
                PACKAGE_ROOT.as_posix(),
            ],
            cwd=PACKAGE_ROOT,
        )
        wheel = _only_artifact(artifact_directory, "*.whl")
        sdist = _only_artifact(artifact_directory, "*.tar.gz")
        _run(
            [sys.executable, "-m", "twine", "check", wheel.as_posix(), sdist.as_posix()],
            cwd=PACKAGE_ROOT,
        )
        _assert_artifact_contents(wheel, sdist)

        for artifact in (wheel, sdist):
            _verify_installed_artifact(
                artifact=artifact,
                expected_version=expected_version,
                fixture=ECHO_SERVER,
                work_directory=work_directory,
            )

    print(f"PASS: wheel and sdist for dendrux {expected_version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed-check", action="store_true")
    parser.add_argument("--expected-version")
    parser.add_argument("--fixture", type=Path)
    arguments = parser.parse_args()

    if arguments.installed_check:
        if arguments.expected_version is None or arguments.fixture is None:
            parser.error("--installed-check requires --expected-version and --fixture")
        _installed_check(expected_version=arguments.expected_version, fixture=arguments.fixture)
        return
    _build_and_verify()


if __name__ == "__main__":
    main()
