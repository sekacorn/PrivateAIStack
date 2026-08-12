from pathlib import Path

import pytest

from private_ai_stack.reviews.collector import collect_repository, materialize_snapshot
from private_ai_stack.reviews.findings import ToolRun
from private_ai_stack.reviews.normalizers import normalize_tool_runs
from private_ai_stack.reviews.reports import markdown_report
from private_ai_stack.tools.command_runner import CommandRunner


def test_collect_repository_excludes_secret_paths(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    snapshot = collect_repository(str(tmp_path), 250_000)

    assert [file.relative_path for file in snapshot.files] == ["app.py"]
    assert ".env" in snapshot.excluded


def test_collect_repository_excludes_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked-secret.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available in this environment")

    snapshot = collect_repository(str(tmp_path), 250_000)

    assert "linked-secret.txt" not in [file.relative_path for file in snapshot.files]
    assert snapshot.excluded["linked-secret.txt"] == "symlink"


def test_normalize_unavailable_tool() -> None:
    findings = normalize_tool_runs([ToolRun(tool="ruff", status="not_run", reason="missing")])
    assert findings[0].status == "open"
    assert findings[0].severity == "informational"
    assert "did not run" in findings[0].title


def test_normalize_ruff_and_bandit_json() -> None:
    ruff = ToolRun(
        tool="ruff",
        status="failed",
        stdout='[{"filename":"app.py","location":{"row":2},"code":"F401","message":"unused import"}]',
        exit_code=1,
    )
    bandit = ToolRun(
        tool="bandit",
        status="failed",
        stdout='{"results":[{"filename":"app.py","line_number":3,"test_id":"B602","issue_text":"shell=True","issue_severity":"HIGH","issue_confidence":"HIGH"}]}',
        exit_code=1,
    )
    findings = normalize_tool_runs([ruff, bandit])
    assert {finding.source_tool for finding in findings} == {"ruff", "bandit"}
    assert any(finding.severity == "high" for finding in findings)


def test_markdown_report_contains_required_sections() -> None:
    summary = {
        "files_reviewed": 1,
        "files_excluded": 0,
        "finding_count": 0,
        "tools": [{"tool": "ruff", "status": "not_run", "reason": "missing"}],
    }
    content = markdown_report(summary, [])
    assert "Technical Quality and Risk" in content
    assert "Tools That Did Not Run" in content


def test_collect_repository_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    try:
        collect_repository(str(missing), 10)
    except ValueError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing repository should fail")


def test_materialized_snapshot_excludes_secret_content(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    snapshot = collect_repository(str(tmp_path), 250_000)
    staged = tmp_path / "staged"

    materialize_snapshot(snapshot, staged)

    assert (staged / "app.py").exists()
    assert not (staged / ".env").exists()


def test_command_runner_bounds_flooding_output() -> None:
    runner = CommandRunner(timeout_seconds=5, max_output_bytes=256)

    run = runner.run("python", ["-c", "print('x' * 10000)"], Path.cwd(), "Python is required.")

    assert run.status == "failed"
    assert run.reason == "output_limit_exceeded"
    assert len(run.stdout.encode("utf-8")) <= 256


def test_command_runner_times_out() -> None:
    runner = CommandRunner(timeout_seconds=0.05, max_output_bytes=256)

    run = runner.run("python", ["-c", "import time; time.sleep(1)"], Path.cwd(), "Python is required.")

    assert run.status == "failed"
    assert run.reason == "timeout"
