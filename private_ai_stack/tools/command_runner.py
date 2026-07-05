import os
import shutil
import subprocess  # nosec B404
import tempfile
from pathlib import Path

from private_ai_stack.reviews.findings import ToolRun


class CommandRunner:
    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    def run(self, tool: str, args: list[str], cwd: Path, install_hint: str) -> ToolRun:
        executable = shutil.which(tool)
        if executable is None:
            return ToolRun(tool=tool, status="not_run", reason=f"{tool} is not installed. {install_hint}")
        cache_root = Path(tempfile.gettempdir()) / "private-ai-stack-tool-cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "HOME": str(cache_root),
            "XDG_CACHE_HOME": str(cache_root),
            "MYPY_CACHE_DIR": str(cache_root / "mypy"),
        }
        try:
            # Tool names and arguments are selected by PrivateAIStack's static review pipeline.
            # The reviewed repository does not control the executable path, and shell=True is not used.
            completed = subprocess.run(  # nosec B603
                [executable, *args],
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return ToolRun(
                tool=tool,
                status="passed" if completed.returncode == 0 else "failed",
                stdout=completed.stdout[-20_000:],
                stderr=completed.stderr[-20_000:],
                exit_code=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
            return ToolRun(tool=tool, status="failed", reason="timeout", stdout=stdout, stderr=stderr, exit_code=None)
