import tempfile
import time
from pathlib import Path

from private_ai_stack.reviews.findings import ToolRun
from private_ai_stack.tools.command_runner import CommandRunner


def run_static_tools(root: Path, has_python: bool, total_timeout_seconds: float, max_output_bytes: int) -> list[ToolRun]:
    runner = CommandRunner(timeout_seconds=90, max_output_bytes=max_output_bytes)
    runs: list[ToolRun] = []
    cache_root = Path(tempfile.gettempdir()) / "private-ai-stack-tool-cache"
    deadline = time.monotonic() + total_timeout_seconds

    def run(tool: str, args: list[str], install_hint: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            runs.append(ToolRun(tool=tool, status="failed", reason="review_timeout", exit_code=None))
            return
        runner.timeout_seconds = min(90.0, remaining)
        runs.append(runner.run(tool, args, root, install_hint))

    mypy_config = "[mypy]\nfollow_imports = skip\nno_site_packages = True\n"
    (root / ".private-ai-stack-mypy.ini").write_text(mypy_config, encoding="utf-8")
    if has_python:
        run(
            "ruff",
            [
                "check",
                ".",
                "--isolated",
                "--output-format",
                "json",
                "--cache-dir",
                str(cache_root / "ruff"),
            ],
            "Install with: pip install ruff",
        )
        run(
            "mypy",
            [
                "--config-file",
                ".private-ai-stack-mypy.ini",
                "--cache-dir",
                str(cache_root / "mypy"),
                ".",
            ],
            "Install with: pip install mypy",
        )
        run("bandit", ["-r", ".", "-f", "json"], "Install with: pip install bandit")
        run("radon", ["cc", ".", "-j"], "Install with: pip install radon")
    run("detect-secrets", ["scan", "--all-files"], "Install with: pip install detect-secrets")
    run("yamllint", ["."], "Install with: pip install yamllint")
    run("markdownlint", ["."], "Install markdownlint-cli.")
    shell_files = [str(path.relative_to(root)) for path in root.rglob("*.sh")]
    run("shellcheck", ["-x", *shell_files] if shell_files else ["--version"], "Install ShellCheck.")
    run("hadolint", ["Dockerfile"] if (root / "Dockerfile").exists() else ["--version"], "Install Hadolint.")
    return runs
