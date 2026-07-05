import tempfile
from pathlib import Path

from private_ai_stack.reviews.findings import ToolRun
from private_ai_stack.tools.command_runner import CommandRunner


def run_static_tools(root: Path, has_python: bool) -> list[ToolRun]:
    runner = CommandRunner(timeout_seconds=90)
    runs: list[ToolRun] = []
    cache_root = Path(tempfile.gettempdir()) / "private-ai-stack-tool-cache"
    if has_python:
        runs.append(
            runner.run(
                "ruff",
                ["check", ".", "--output-format", "json", "--cache-dir", str(cache_root / "ruff")],
                root,
                "Install with: pip install ruff",
            )
        )
        runs.append(runner.run("mypy", ["--cache-dir", str(cache_root / "mypy"), "."], root, "Install with: pip install mypy"))
        runs.append(runner.run("bandit", ["-r", ".", "-f", "json"], root, "Install with: pip install bandit"))
        runs.append(runner.run("pip-audit", ["--format", "json"], root, "Install with: pip install pip-audit"))
        runs.append(runner.run("radon", ["cc", ".", "-j"], root, "Install with: pip install radon"))
    runs.append(runner.run("detect-secrets", ["scan", "--all-files"], root, "Install with: pip install detect-secrets"))
    runs.append(runner.run("yamllint", ["."], root, "Install with: pip install yamllint"))
    runs.append(runner.run("markdownlint", ["."], root, "Install markdownlint-cli."))
    shell_files = [str(path.relative_to(root)) for path in root.rglob("*.sh")]
    runs.append(runner.run("shellcheck", ["-x", *shell_files] if shell_files else ["--version"], root, "Install ShellCheck."))
    runs.append(runner.run("hadolint", ["Dockerfile"] if (root / "Dockerfile").exists() else ["--version"], root, "Install Hadolint."))
    return runs
