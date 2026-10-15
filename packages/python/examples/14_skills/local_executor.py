"""A local subprocess executor for skill scripts.

WARNING: this runs skill scripts on your machine, with your user's
permissions. Use it only for skills you wrote or have reviewed. For skills
you do not control, run scripts in a container or a hosted sandbox instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import tempfile
from pathlib import Path

from dendrux import SkillExecutionRequest, SkillExecutionResult, SkillExecutor

INTERPRETERS = {".py": sys.executable, ".sh": "/bin/sh"}


class LocalSubprocessExecutor(SkillExecutor):
    """Runs skill scripts as subprocesses in a throwaway directory."""

    timeout_seconds = 60.0

    async def execute(self, request: SkillExecutionRequest) -> SkillExecutionResult:
        with tempfile.TemporaryDirectory(prefix="skill-") as tmp:
            root = Path(tmp)
            for path, content in request.files.items():
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            script = root / request.script_path
            interpreter = INTERPRETERS.get(script.suffix)
            if interpreter is None:
                return SkillExecutionResult(
                    stderr=f"Unsupported script type: {script.suffix}", exit_code=126
                )

            proc = await asyncio.create_subprocess_exec(
                interpreter,
                str(script),
                *request.args,
                cwd=root,
                env={"PATH": os.environ.get("PATH", "")},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # own process group, so we can kill descendants too
            )
            try:
                stdout, stderr = await proc.communicate()
            except BaseException:  # timed out or the task was cancelled
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
                raise

            return SkillExecutionResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                exit_code=proc.returncode or 0,
            )
