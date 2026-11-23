"""The ``privateaistack reembed`` commands: ``run``, ``--estimate-only`` and ``quality``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from private_ai_stack.cli.main import app
from tests.unit.reembed_support import build_ranked_store, build_source_store

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


# --- reembed quality --------------------------------------------------------


@pytest.fixture
def ranked(tmp_path: Path) -> tuple[Path, Path]:
    """An original store and its rebuild, both with real vectors on disk."""
    before = build_ranked_store(tmp_path / "before")
    result = runner.invoke(
        app,
        [
            "reembed",
            "run",
            str(before),
            "--source-store",
            "old-store",
            "--target-directory",
            str(tmp_path / "after"),
            "--target-store",
            "new-store",
            "--model-id",
            "privateaistack.hashed-bow.sha256-signed",
            "--model-revision",
            "1",
            "--dimensions",
            "64",
            "--availability",
            "open_weight_self_hostable",
        ],
    )
    assert result.exit_code == 0, result.output
    return before, tmp_path / "after" / "embedding-portability.json"


def _quality(before: Path, after: Path, *extra: str) -> Any:
    return runner.invoke(
        app,
        [
            "reembed",
            "quality",
            str(before),
            str(after),
            "--before-store",
            "old-store",
            "--after-store",
            "new-store",
            *extra,
        ],
    )


def test_the_quality_command_reports_a_seeded_record_anchored_comparison(ranked: tuple[Path, Path]) -> None:
    before, after = ranked

    result = _quality(before, after, "--top-k", "5", "--min-top-k-overlap", "0.1", "--anchors", "24")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "sampled"
    assert payload["judged"] is True
    assert payload["anchor_selection"] == "seeded_random"
    assert payload["metric_basis"] == "neighborhood_agreement"
    assert payload["measures_user_search_quality"] is False
    assert payload["moves_lock_in_score"] is True


def test_an_operator_query_set_is_reported_as_moving_no_score(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    before, after = ranked
    query_set = tmp_path / "set.json"
    query_set.write_text(
        json.dumps(
            {
                "schema_version": "golden-query-set.v1",
                "mode": "record_anchored",
                "anchors": [f"r{index:04d}" for index in range(24)],
            }
        ),
        encoding="utf-8",
    )

    result = _quality(before, after, "--query-set", str(query_set), "--top-k", "5", "--min-top-k-overlap", "0.1")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["anchor_selection"] == "operator"
    assert payload["moves_lock_in_score"] is False
    assert "anchor_seed" not in payload


def test_the_quality_command_reports_unjudged_without_a_declared_floor(
    ranked: tuple[Path, Path],
) -> None:
    before, after = ranked

    result = _quality(before, after, "--top-k", "5", "--anchors", "24", "--min-sample-size", "10")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "unjudged"
    assert payload["judged"] is False
    assert payload["moves_lock_in_score"] is False
    # The numbers survive, and the output says what to declare for a verdict.
    assert "top_k_overlap" in payload["metrics"]
    assert "minimum_top_k_overlap" in payload["notes"]


def test_an_empty_draw_reports_not_measured_rather_than_failing(ranked: tuple[Path, Path]) -> None:
    before, after = ranked

    result = _quality(before, after, "--anchors", "0", "--min-top-k-overlap", "0.1")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "not_measured"
    assert payload["moves_lock_in_score"] is False


def test_the_quality_command_writes_into_the_re_embed_record(ranked: tuple[Path, Path], tmp_path: Path) -> None:
    before, after = ranked
    record = tmp_path / "after" / "re-embed-record.json"

    result = _quality(before, after, "--top-k", "5", "--min-top-k-overlap", "0.1", "--anchors", "24", "--record", str(record))

    assert result.exit_code == 0, result.output
    document = json.loads(record.read_text(encoding="utf-8"))
    assert document["quality"]["status"] == "sampled"
    assert document["quality"]["anchor_seed"] == "re-embed-quality"


def test_an_unknown_store_exits_two(ranked: tuple[Path, Path]) -> None:
    before, after = ranked

    result = _quality(before, after, "--min-top-k-overlap", "0.1")
    assert result.exit_code == 0

    bad = runner.invoke(
        app,
        [
            "reembed",
            "quality",
            str(before),
            str(after),
            "--before-store",
            "nosuchstore",
            "--after-store",
            "new-store",
            "--min-top-k-overlap",
            "0.1",
        ],
    )
    assert bad.exit_code == 2


def test_the_quality_command_is_discoverable() -> None:
    result = runner.invoke(app, ["reembed", "--help"])

    assert result.exit_code == 0
    assert "quality" in result.output


# --- egress flags -----------------------------------------------------------


def _egress_run(source: Path, target: Path, *extra: str) -> Any:
    return runner.invoke(app, _arguments(source, target) + list(extra))


def test_a_run_is_local_unless_told_otherwise(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target")

    assert result.exit_code == 0, result.output
    record = json.loads((tmp_path / "target" / "re-embed-record.json").read_text(encoding="utf-8"))
    assert record["egress"]["hosting_mode"] == "local"
    assert record["egress"]["remote_provider_used"] is False


def test_a_hosted_run_without_allow_hosted_exits_two(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target", "--hosting-mode", "hosted", "--provider", "example-hosted")

    assert result.exit_code == 2
    assert "Hosted execution was not enabled" in result.output
    assert not (tmp_path / "target" / "re-embed-record.json").exists()


def test_the_blocking_warning_is_shown_before_the_refusal(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target", "--hosting-mode", "hosted", "--provider", "example-hosted")

    assert "hosted execution is blocked until explicitly enabled" in result.output


def test_an_allowed_hosted_run_from_the_cli_says_no_provider_ships(source: Path, tmp_path: Path) -> None:
    # The CLI cannot supply a hosted embedder, so this is where an operator
    # learns the boundary rather than getting a silent local run.
    result = _egress_run(
        source,
        tmp_path / "target",
        "--hosting-mode",
        "hosted",
        "--provider",
        "example-hosted",
        "--allow-hosted",
    )

    assert result.exit_code == 2
    assert "ships no hosted provider" in result.output


def test_a_local_run_naming_a_provider_exits_two(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target", "--provider", "example-hosted")

    assert result.exit_code == 2
    assert "must not name a provider" in result.output


def test_an_unknown_classification_exits_two(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target", "--data-classification", "top-secret")

    assert result.exit_code == 2


def test_telemetry_is_warned_about_and_recorded(source: Path, tmp_path: Path) -> None:
    result = _egress_run(source, tmp_path / "target", "--telemetry-enabled")

    assert result.exit_code == 0, result.output
    assert "telemetry is enabled" in result.output
    record = json.loads((tmp_path / "target" / "re-embed-record.json").read_text(encoding="utf-8"))
    assert record["egress"]["telemetry_enabled"] is True
    assert record["egress"]["warnings"] == ["telemetry is enabled; review exported fields before execution"]
