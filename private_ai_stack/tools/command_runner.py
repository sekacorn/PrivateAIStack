import os
import shutil
import subprocess  # nosec B404
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

from private_ai_stack.reviews.findings import ToolRun


class CommandRunner:
    """Run selected static-analysis tools without a shell and with bounded capture."""

    def __init__(self, timeout_seconds: float = 60.0, max_output_bytes: int = 20_000) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

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
            process = subprocess.Popen(  # nosec B603
                [executable, *args],
                cwd=cwd,
                env=env,
                text=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            return ToolRun(tool=tool, status="not_run", reason=f"{tool} could not start: {exc.__class__.__name__}. {install_hint}")

        captured: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        output_limit_hit = threading.Event()

        def drain(name: str, reader: BinaryIO | None) -> None:
            if reader is None:
                return
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    return
                remaining = self.max_output_bytes - len(captured[name])
                if remaining > 0:
                    captured[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    output_limit_hit.set()

        threads = [
            threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while process.poll() is None:
                if output_limit_hit.is_set():
                    process.kill()
                    process.wait()
                    return ToolRun(
                        tool=tool,
                        status="failed",
                        reason="output_limit_exceeded",
                        stdout=captured["stdout"].decode("utf-8", errors="replace"),
                        stderr=captured["stderr"].decode("utf-8", errors="replace"),
                        exit_code=process.returncode,
                    )
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired([executable, *args], self.timeout_seconds)
                time.sleep(0.02)
            if output_limit_hit.is_set():
                return ToolRun(
                    tool=tool,
                    status="failed",
                    reason="output_limit_exceeded",
                    stdout=captured["stdout"].decode("utf-8", errors="replace"),
                    stderr=captured["stderr"].decode("utf-8", errors="replace"),
                    exit_code=process.returncode,
                )
            return ToolRun(
                tool=tool,
                status="passed" if process.returncode == 0 else "failed",
                stdout=captured["stdout"].decode("utf-8", errors="replace"),
                stderr=captured["stderr"].decode("utf-8", errors="replace"),
                exit_code=process.returncode,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return ToolRun(
                tool=tool,
                status="failed",
                reason="timeout",
                stdout=captured["stdout"].decode("utf-8", errors="replace"),
                stderr=captured["stderr"].decode("utf-8", errors="replace"),
                exit_code=None,
            )
        finally:
            for thread in threads:
                thread.join(timeout=1)
