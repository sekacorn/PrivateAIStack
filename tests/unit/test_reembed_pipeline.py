"""The offline re-embedding pipeline: what it produces, and what it refuses.

The invariant these tests exist for is not asserted anywhere in the code: after
any number of interruptions at any point, the finished store holds every
retained record exactly once, in a deterministic order, with no duplicates and
no omissions. It is demonstrated here by interrupting a run at every point one
can be interrupted and comparing the result, byte for byte, with a run that was
never interrupted.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from reembed_support import build_source_store, tree_digest, write_document

from private_ai_stack.reembed.checkpoint import CHECKPOINT_FILENAME, load_checkpoint
from private_ai_stack.reembed.embedders import PRIVATEAISTACK_HASHED_BOW, Embedder, ModelIdentity, build_built_in_embedder
from private_ai_stack.reembed.pipeline import (
    DEFAULT_BATCH_SIZE,
    MAX_SHARD_LINE_BYTES,
    ReEmbedError,
    ReEmbedOutcome,
    re_embed_store,
)

TARGET = ModelIdentity(id=PRIVATEAISTACK_HASHED_BOW, revision="1", dimensions=16)
STORE_ONLY = ("checkpoint.json", "re-embed-record.json")


def many_texts(count: int) -> dict[str, str]:
    return {f"record-{index:03d}": f"Document number {index} about refunds, escalation and passwords." for index in range(count)}


def run(
    source: Path,
    target: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedder: Embedder | None = None,
    model: ModelIdentity = TARGET,
    availability: str = "open_weight_self_hostable",
    **extra: Any,
) -> ReEmbedOutcome:
    return re_embed_store(
        source_document=source,
        source_store_id="support-memory",
        target_directory=target,
        target_store_id="support-memory-rebuilt",
        target_model=model,
        availability=availability,
        batch_size=batch_size,
        embedder=embedder,
        **extra,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class CountingEmbedder:
    """A real embedder that stops after ``limit`` calls, to simulate a crash."""

    def __init__(self, limit: int, model: ModelIdentity = TARGET) -> None:
        self._inner = build_built_in_embedder(model)
        self._limit = limit
        self.calls = 0

    @property
    def identity(self) -> ModelIdentity:
        return self._inner.identity

    def embed(self, text: str) -> list[float]:
        if self.calls >= self._limit:
            raise KeyboardInterrupt("simulated interruption")
        self.calls += 1
        return self._inner.embed(text)


class WrongWidthEmbedder:
    """Declares the target identity but produces a vector of the wrong width."""

    def __init__(self, width: int) -> None:
        self._width = width

    @property
    def identity(self) -> ModelIdentity:
        return TARGET

    def embed(self, text: str) -> list[float]:
        return [0.5] * self._width


class MislabelledEmbedder:
    """Reports an identity other than the one the run declares."""

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(id="some-vendor-embedding-v3", revision="1", dimensions=16)

    def embed(self, text: str) -> list[float]:
        return [0.0] * 16


# --- what a completed run produces ------------------------------------------------


def test_a_run_produces_a_store_a_record_and_a_checkpoint(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    outcome = run(source, tmp_path / "target", batch_size=2)
    assert outcome.target_document.is_file()
    assert outcome.record.is_file()
    assert outcome.checkpoint.is_file()
    assert outcome.retained_rebuilt == 3
    assert outcome.batches_completed == 2
    assert outcome.complete


def test_the_rebuilt_store_is_an_ordinary_evidence_document(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    outcome = run(source, tmp_path / "target", batch_size=2)
    document = read_json(outcome.target_document)
    store = document["stores"][0]
    assert document["schema_version"] == "embedding-portability.v1"
    assert store["store_id"] == "support-memory-rebuilt"
    assert store["embedding_model"]["dimensions"] == 16
    assert store["embedding_model"]["availability_verification"] == "attested"
    assert store["vectors"]["layout"] == "sharded_jsonl"
    assert [shard["record_count"] for shard in store["vectors"]["shards"]] == [2, 1]
    assert len(store["vectors"]["source_text"]) == 3


def test_every_declared_digest_in_the_rebuilt_store_matches_the_file_on_disk(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    target = tmp_path / "target"
    outcome = run(source, target, batch_size=2)
    store = read_json(outcome.target_document)["stores"][0]
    for shard in store["vectors"]["shards"]:
        assert hashlib.sha256((target / shard["path"]).read_bytes()).hexdigest() == shard["sha256"]
    for entry in store["vectors"]["source_text"]:
        text = entry["source_text"]
        assert hashlib.sha256((target / text["path"]).read_bytes()).hexdigest() == text["sha256"]


def test_each_shard_line_carries_the_source_digest_its_vector_came_from(tmp_path: Path) -> None:
    """The line shape ``embedding-portability.v1`` streams: nothing more, nothing less.

    A line holding only values would leave a reader unable to check that the
    text beside the store is the text the store was built from, which is the
    whole point of retaining it.
    """
    source = build_source_store(tmp_path / "source")
    target = tmp_path / "target"
    outcome = run(source, target, batch_size=2)
    declared = {
        entry["record_id"]: entry["source_text"]["sha256"]
        for entry in read_json(outcome.target_document)["stores"][0]["vectors"]["source_text"]
    }
    lines = [json.loads(line) for path in sorted((target / "vectors").iterdir()) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines, "the run wrote no vectors"
    for entry in lines:
        assert set(entry) == {"record_id", "source_text_sha256", "values"}
        assert entry["source_text_sha256"] == declared[entry["record_id"]]


def test_the_vectors_are_the_ones_the_declared_model_produces(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    expected = build_built_in_embedder(TARGET)
    lines = [json.loads(line) for path in sorted((target / "vectors").iterdir()) for line in path.read_text(encoding="utf-8").splitlines()]
    identifiers = [entry["record_id"] for entry in lines]
    assert identifiers == sorted(identifiers), "vectors are written in a deterministic order"
    sources = {
        entry["record_id"]: entry["source_text"]["path"]
        for entry in read_json(target / "embedding-portability.json")["stores"][0]["vectors"]["source_text"]
    }
    for entry in lines:
        text = (target / sources[entry["record_id"]]).read_text(encoding="utf-8")
        assert entry["values"] == expected.embed(text)


def test_the_record_accounts_for_the_run(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", missing=("gone",), deleted_by_policy=("purged",))
    target = tmp_path / "target"
    outcome = run(source, target, batch_size=2)
    record = read_json(outcome.record)
    assert record["schema_version"] == "re-embed-record.v1"
    assert record["source_store"]["store_evidence_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["target_store"]["store_evidence_sha256"] == hashlib.sha256(outcome.target_document.read_bytes()).hexdigest()
    assert record["coverage"] == {
        "total_records": 5,
        "retained_rebuilt": 3,
        "missing_skipped": 1,
        "deleted_by_policy_skipped": 1,
        "failed": 0,
    }
    assert record["batch_log"] == {
        "batch_size": 2,
        "batches_completed": 2,
        "records_embedded": 3,
        "resumed": False,
        "checkpoint_sha256": hashlib.sha256(outcome.checkpoint.read_bytes()).hexdigest(),
    }
    assert record["egress"] == {"remote_provider_used": False, "records_transmitted": 0, "raw_source_text_transmitted": False}


def test_the_record_makes_no_quality_claim_because_nothing_was_compared(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    record = read_json(run(source, tmp_path / "target").record)
    assert "quality" not in record


def test_the_attested_model_block_is_carried_into_both_documents(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    outcome = run(source, tmp_path / "target", weights_sha256="a" * 64, license_name="Apache-2.0")
    model = read_json(outcome.record)["target_model"]
    assert model["weights_sha256"] == "a" * 64
    assert model["license"] == "Apache-2.0"
    assert read_json(outcome.target_document)["stores"][0]["embedding_model"] == model


# --- records with no text -----------------------------------------------------------


def test_records_with_no_text_are_never_embedded_and_never_appear_in_the_rebuilt_store(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", missing=("gone",), deleted_by_policy=("purged",))
    target = tmp_path / "target"
    outcome = run(source, target)
    store = read_json(outcome.target_document)["stores"][0]
    identifiers = {entry["record_id"] for entry in store["vectors"]["source_text"]}
    assert "purged" not in identifiers
    assert "gone" not in identifiers
    body = "\n".join(path.read_text(encoding="utf-8") for path in (target / "vectors").iterdir())
    assert "purged" not in body
    assert "gone" not in body
    assert outcome.deleted_by_policy_skipped == 1


def test_a_store_whose_records_all_lack_text_is_refused_rather_than_rebuilt_empty(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts={}, deleted_by_policy=("purged",))
    with pytest.raises(ReEmbedError, match="no records|no retained records"):
        run(source, tmp_path / "target")


# --- idempotence and resume ----------------------------------------------------------


def test_rerunning_a_finished_job_changes_no_byte_of_the_store(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(7))
    target = tmp_path / "target"
    first = run(source, target, batch_size=3)
    before = tree_digest(target)
    second = run(source, target, batch_size=3)
    assert tree_digest(target, skip=STORE_ONLY) == {k: v for k, v in before.items() if k not in STORE_ONLY}
    assert second.batches_embedded_now == 0
    assert second.resumed is True
    assert first.retained_rebuilt == second.retained_rebuilt == 7


def test_rerunning_changes_only_the_resumed_flag_in_the_record(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(5))
    target = tmp_path / "target"
    first = read_json(run(source, target, batch_size=2).record)
    second = read_json(run(source, target, batch_size=2).record)
    assert first["batch_log"].pop("resumed") is False
    assert second["batch_log"].pop("resumed") is True
    assert first == second


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 6])
def test_an_interrupted_run_finishes_to_the_same_bytes_as_an_uninterrupted_one(tmp_path: Path, limit: int) -> None:
    """Negative test 1: crash and resume, at every point a crash can land."""
    source = build_source_store(tmp_path / "source", texts=many_texts(7))
    reference = tmp_path / "reference"
    run(source, reference, batch_size=2)
    expected = tree_digest(reference, skip=STORE_ONLY)

    target = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        run(source, target, batch_size=2, embedder=CountingEmbedder(limit))
    assert (target / CHECKPOINT_FILENAME).exists() or limit < 2

    outcome = run(source, target, batch_size=2)
    assert tree_digest(target, skip=STORE_ONLY) == expected
    assert outcome.retained_rebuilt == 7
    assert outcome.resumed is (limit >= 2)


def test_resume_does_not_re_embed_what_is_already_done(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(6))
    target = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        run(source, target, batch_size=2, embedder=CountingEmbedder(4))
    counter = CountingEmbedder(100)
    run(source, target, batch_size=2, embedder=counter)
    assert counter.calls == 2, "only the one unfinished batch should be re-embedded"


def test_every_retained_record_appears_exactly_once_after_an_interruption(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(9))
    target = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        run(source, target, batch_size=4, embedder=CountingEmbedder(5))
    run(source, target, batch_size=4)
    written = [
        json.loads(line)["record_id"]
        for path in sorted((target / "vectors").iterdir())
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == sorted(many_texts(9))
    assert len(written) == len(set(written)) == 9


def test_a_shard_no_checkpoint_claims_is_discarded_rather_than_trusted(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    orphan = target / "vectors" / "shard-0007.jsonl"
    orphan.write_text('{"record_id": "fabricated", "values": [0.0]}\n', encoding="utf-8")
    run(source, target, batch_size=2)
    assert not orphan.exists()


def test_a_file_this_pipeline_did_not_name_is_left_alone(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    bystander = target / "vectors" / "operator-notes.txt"
    bystander.write_text("do not delete", encoding="utf-8")
    run(source, target, batch_size=2)
    assert bystander.read_text(encoding="utf-8") == "do not delete"


# --- corrupted or mismatched checkpoints ---------------------------------------------


def corrupt(target: Path, mutate: Any) -> None:
    document = read_json(target / CHECKPOINT_FILENAME)
    mutate(document)
    write_document(target, document, name=CHECKPOINT_FILENAME)


def test_a_truncated_checkpoint_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    raw = (target / CHECKPOINT_FILENAME).read_text(encoding="utf-8")
    (target / CHECKPOINT_FILENAME).write_text(raw[: len(raw) // 2], encoding="utf-8")
    with pytest.raises(ReEmbedError, match="not valid JSON"):
        run(source, target, batch_size=2)


def test_a_checkpoint_with_a_mutated_shard_digest_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    corrupt(target, lambda document: document["batches"][0].__setitem__("shard_sha256", "b" * 64))
    with pytest.raises(ReEmbedError, match="Something other than this run changed it"):
        run(source, target, batch_size=2)


def test_a_checkpoint_with_a_mutated_record_list_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    corrupt(target, lambda document: document["batches"][0].__setitem__("record_ids", ["record-001", "record-000"]))
    with pytest.raises(ReEmbedError, match="would not have placed there"):
        run(source, target, batch_size=2)


def test_a_checkpoint_written_against_different_input_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    corrupt(target, lambda document: document.__setitem__("source_evidence_sha256", "c" * 64))
    with pytest.raises(ReEmbedError, match="different input document"):
        run(source, target, batch_size=2)


def test_editing_the_input_after_a_partial_run_is_refused(tmp_path: Path) -> None:
    """The realistic form of the same failure: the operator changes the store mid-run."""
    source_dir = tmp_path / "source"
    source = build_source_store(source_dir, texts=many_texts(6))
    target = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        run(source, target, batch_size=2, embedder=CountingEmbedder(2))
    build_source_store(source_dir, texts=many_texts(8))
    with pytest.raises(ReEmbedError, match="different input document"):
        run(source, target, batch_size=2)


def test_a_checkpoint_claiming_a_shard_that_is_not_there_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    (target / "vectors" / "shard-0000.jsonl").unlink()
    with pytest.raises(ReEmbedError, match="not on disk"):
        run(source, target, batch_size=2)


def test_a_checkpoint_claiming_the_same_record_twice_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    corrupt(target, lambda document: document["batches"][1].__setitem__("record_ids", document["batches"][0]["record_ids"]))
    with pytest.raises(ReEmbedError, match="same record in more than one batch"):
        run(source, target, batch_size=2)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", "re-embed-checkpoint.v2", "Expected re-embed-checkpoint.v1"),
        ("batch_size", 0, "not a positive integer"),
        ("batches", {}, "no batches array"),
        ("target_model", [], "no target_model object"),
        ("target_store_id", "", "no target_store_id"),
    ],
)
def test_a_structurally_unusable_checkpoint_is_refused(tmp_path: Path, key: str, value: Any, message: str) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    corrupt(target, lambda document: document.__setitem__(key, value))
    with pytest.raises(ReEmbedError, match=message):
        run(source, target, batch_size=2)


def test_resuming_with_a_different_batch_size_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    with pytest.raises(ReEmbedError, match="used batch size 2, not 3"):
        run(source, target, batch_size=3)


def test_resuming_with_a_different_model_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", texts=many_texts(4))
    target = tmp_path / "target"
    run(source, target, batch_size=2)
    wider = ModelIdentity(id=PRIVATEAISTACK_HASHED_BOW, revision="1", dimensions=32)
    with pytest.raises(ReEmbedError, match="is for model"):
        run(source, target, batch_size=2, model=wider)


def test_the_shipped_checkpoint_fixture_still_loads(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "reembed" / "re-embed-checkpoint.json"
    checkpoint = load_checkpoint(fixture)
    assert checkpoint.batch_size == 2
    assert len(checkpoint.batches) == 2
    assert checkpoint.completed_record_ids == ("record-000", "record-001", "record-002", "record-003")


def test_the_shipped_fixture_carries_every_key_the_schema_requires() -> None:
    root = Path(__file__).resolve().parents[2]
    schema = json.loads((root / "private_ai_stack" / "reembed" / "schemas" / "re-embed-checkpoint.schema.json").read_text(encoding="utf-8"))
    fixture = json.loads((root / "tests" / "fixtures" / "reembed" / "re-embed-checkpoint.json").read_text(encoding="utf-8"))
    assert set(schema["required"]) <= set(fixture)
    assert set(fixture) <= set(schema["properties"])
    batch_schema = schema["properties"]["batches"]["items"]
    for batch in fixture["batches"]:
        assert set(batch_schema["required"]) <= set(batch)


# --- identity and declaration mismatches ---------------------------------------------


def test_an_embedder_that_produces_the_wrong_width_stops_the_run(tmp_path: Path) -> None:
    """Negative test 3: dimension mismatch, rejected before the store is written."""
    source = build_source_store(tmp_path / "source")
    target = tmp_path / "target"
    with pytest.raises(ReEmbedError, match="produced 8 dimensions.*declared model is 16"):
        run(source, target, embedder=WrongWidthEmbedder(8))
    assert not (target / "embedding-portability.json").exists()
    assert not (target / CHECKPOINT_FILENAME).exists()


def test_an_embedder_whose_identity_is_not_the_declared_one_stops_the_run(tmp_path: Path) -> None:
    """Negative test 7: the model that produced the vectors is not the declared model."""
    source = build_source_store(tmp_path / "source")
    with pytest.raises(ReEmbedError, match="does not match the declared target"):
        run(source, tmp_path / "target", embedder=MislabelledEmbedder())


def test_an_unknown_target_model_stops_the_run(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    unknown = ModelIdentity(id="some-vendor-embedding-v3", revision="1", dimensions=16)
    with pytest.raises(ReEmbedError, match="not a built-in embedder"):
        run(source, tmp_path / "target", model=unknown)


def test_an_availability_the_format_does_not_define_is_refused(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source")
    with pytest.raises(ReEmbedError, match="availability must be one of"):
        run(source, tmp_path / "target", availability="probably_fine")


@pytest.mark.parametrize("batch_size", [0, -1, 10_001])
def test_a_batch_size_outside_the_bound_is_refused(tmp_path: Path, batch_size: int) -> None:
    source = build_source_store(tmp_path / "source")
    with pytest.raises(ReEmbedError, match="batch_size must be between"):
        run(source, tmp_path / "target", batch_size=batch_size)


def test_a_vector_too_wide_for_one_jsonl_line_is_refused(tmp_path: Path) -> None:
    """A dense vector at the format's widest dimension does not fit one line, so the run stops."""
    dense = " ".join(f"token{index}" for index in range(6000))
    source = build_source_store(tmp_path / "source", texts={"r1": dense})
    wide = ModelIdentity(id=PRIVATEAISTACK_HASHED_BOW, revision="1", dimensions=8192)
    with pytest.raises(ReEmbedError, match=f"over the {MAX_SHARD_LINE_BYTES} byte line limit"):
        run(source, tmp_path / "target", model=wide)


def test_an_absent_source_store_is_reported_not_rebuilt(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "source", store_id="other-memory")
    with pytest.raises(ReEmbedError, match="No store 'support-memory'"):
        run(source, tmp_path / "target")


# --- offline ------------------------------------------------------------------------


def test_a_run_opens_no_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3b-2 is local only. A remote provider arrives in 3b-5, behind an opt-in."""

    def refuse(*args: Any, **kwargs: Any) -> Iterator[None]:
        raise AssertionError("the offline pipeline opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    source = build_source_store(tmp_path / "source", texts=many_texts(5))
    assert run(source, tmp_path / "target", batch_size=2).retained_rebuilt == 5


def test_the_default_batch_size_is_bounded_for_a_small_machine() -> None:
    assert DEFAULT_BATCH_SIZE == 64
