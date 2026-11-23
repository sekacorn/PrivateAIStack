"""The ``privateaistack reembed run`` command, including ``--estimate-only``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from private_ai_stack.cli.main import app
from tests.unit.reembed_support import build_source_store

runner = CliRunner()

MODEL = ("privateaistack.hashed-bow.sha256-signed", "1", "16")
TEXTS = {
    "procedure-escalation": "Escalate to the on-call engineer when the queue depth exceeds 40.",
    "procedure-password-reset": "Reset the password through the self-service portal.",
    "procedure-refund": "Refunds over 500 need a supervisor approval and an audit entry.",
}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return build_source_store(tmp_path / "evidence", texts=TEXTS, dimensions=8)


def _arguments(source: Path, target: Path) -> list[str]:
    return [
        "reembed",
        "run",
        str(source),
        "--source-store",
        "support-memory",
        "--target-directory",
        str(target),
        "--target-store",
        "support-memory-rebuilt",
        "--model-id",
        MODEL[0],
        "--model-revision",
        MODEL[1],
        "--dimensions",
        MODEL[2],
        "--availability",
        "open_weight_self_hostable",
    ]


def test_estimate_only_writes_no_store(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "rebuilt"
    result = runner.invoke(app, [*_arguments(source, target), "--estimate-only", "--sample-size", "2"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["schema_version"] == "re-embed-estimate.v1"
    assert document["records_to_embed"] == len(TEXTS)
    assert document["calibration"]["sample_size"] == 2
    assert not target.exists(), "an estimate must not rebuild the store"


def test_estimate_only_can_also_write_the_document(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "estimate.json"
    result = runner.invoke(
        app,
        [*_arguments(source, tmp_path / "rebuilt"), "--estimate-only", "--sample-size", "1", "--estimate-output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(result.output)


def test_estimate_only_reports_no_money_without_a_profile(source: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, [*_arguments(source, tmp_path / "rebuilt"), "--estimate-only"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["cost"]["status"] == "not_estimated"


def test_a_zero_sample_reports_no_estimate(source: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, [*_arguments(source, tmp_path / "rebuilt"), "--estimate-only", "--sample-size", "0"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["calibration"] is None
    assert document["projection"] is None


def test_a_run_rebuilds_the_store_and_reports_its_coverage(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "rebuilt"
    result = runner.invoke(app, _arguments(source, target))
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["retained_rebuilt"] == len(TEXTS)
    assert document["complete"] is True
    assert document["resumed"] is False
    assert (target / "embedding-portability.json").is_file()
    assert (target / "re-embed-record.json").is_file()


def test_a_second_run_reports_that_it_resumed(source: Path, tmp_path: Path) -> None:
    target = tmp_path / "rebuilt"
    assert runner.invoke(app, _arguments(source, target)).exit_code == 0
    result = runner.invoke(app, _arguments(source, target))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["resumed"] is True


def test_an_unknown_model_exits_with_the_invalid_input_code(source: Path, tmp_path: Path) -> None:
    arguments = _arguments(source, tmp_path / "rebuilt")
    arguments[arguments.index("--model-id") + 1] = "not-a-model"
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2
    assert "not a built-in embedder" in result.output


def test_a_bad_availability_exits_with_the_invalid_input_code(source: Path, tmp_path: Path) -> None:
    arguments = _arguments(source, tmp_path / "rebuilt")
    arguments[arguments.index("--availability") + 1] = "probably-fine"
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2


def test_an_absent_source_document_exits_with_the_invalid_input_code(tmp_path: Path) -> None:
    result = runner.invoke(app, _arguments(tmp_path / "absent.json", tmp_path / "rebuilt"))
    assert result.exit_code == 2


def test_an_absent_cost_profile_exits_with_the_invalid_input_code(source: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [*_arguments(source, tmp_path / "rebuilt"), "--estimate-only", "--cost-profile", str(tmp_path / "absent.json")],
    )
    assert result.exit_code == 2
    assert "cost profile" in result.output.lower()


def test_the_command_is_discoverable(source: Path) -> None:
    result = runner.invoke(app, ["reembed", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output
