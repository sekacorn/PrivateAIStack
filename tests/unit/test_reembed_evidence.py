"""Reading operator-supplied evidence, including everything it must refuse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reembed_support import build_source_store, write_document

from private_ai_stack.reembed.evidence import (
    MAX_SOURCE_TEXT_BYTES,
    EvidenceError,
    load_source_store,
    read_bounded_text,
    reject_unsafe_path,
)


def test_loads_a_store_and_binds_it_to_the_document_bytes(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    store = load_source_store(document, "support-memory")
    assert store.store_id == "support-memory"
    assert store.model_dimensions == 8
    assert store.document_sha256 == __import__("hashlib").sha256(document.read_bytes()).hexdigest()


def test_retained_records_come_back_in_a_deterministic_order(tmp_path: Path) -> None:
    store = load_source_store(build_source_store(tmp_path), "support-memory")
    identifiers = [record.record_id for record in store.retained]
    assert identifiers == sorted(identifiers)


@pytest.mark.parametrize("layout", ["single_file", "sharded_jsonl"])
def test_both_layouts_yield_the_same_records(tmp_path: Path, layout: str) -> None:
    document = build_source_store(tmp_path, layout=layout, missing=("gone",), deleted_by_policy=("purged",))
    store = load_source_store(document, "support-memory")
    assert len(store.records) == 5
    assert len(store.retained) == 3
    assert store.count("missing") == 1
    assert store.count("deleted_by_policy") == 1


def test_reading_a_retained_record_verifies_its_text(tmp_path: Path) -> None:
    store = load_source_store(build_source_store(tmp_path), "support-memory")
    assert "self-service portal" in store.read_text(store.retained[1])


def test_text_that_no_longer_matches_its_digest_stops_the_run(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    (tmp_path / "sources" / "procedure-refund.txt").write_text("Refunds are now unlimited.", encoding="utf-8")
    store = load_source_store(document, "support-memory")
    record = next(item for item in store.retained if item.record_id == "procedure-refund")
    with pytest.raises(EvidenceError, match="source_text.sha256 declares"):
        store.read_text(record)


def test_text_that_disagrees_with_the_vector_record_digest_also_stops_the_run(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    records = payload["stores"][0]["vectors"]["records"]
    records[0]["source_text_sha256_in_vector_record"] = "1" * 64
    write_document(tmp_path, payload)
    store = load_source_store(document, "support-memory")
    with pytest.raises(EvidenceError, match="the vector record declares"):
        store.read_text(store.retained[0])


def test_a_record_with_no_retained_text_cannot_be_read(tmp_path: Path) -> None:
    document = build_source_store(tmp_path, deleted_by_policy=("purged",))
    store = load_source_store(document, "support-memory")
    purged = next(record for record in store.records if record.record_id == "purged")
    with pytest.raises(EvidenceError, match="no retained source text"):
        store.read_text(purged)


@pytest.mark.parametrize(
    "relative",
    ["../escape.txt", "/etc/passwd", "C:/Windows/win.ini", "sources/../../escape.txt", "", "a" * 241, "sources//x.txt"],
)
def test_unsafe_paths_are_refused(tmp_path: Path, relative: str) -> None:
    with pytest.raises(EvidenceError):
        reject_unsafe_path(tmp_path, relative)


def test_a_symbolic_link_is_refused_even_when_it_points_somewhere_harmless(tmp_path: Path) -> None:
    target = tmp_path / "real.txt"
    target.write_text("ordinary", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("creating symbolic links is not permitted in this environment")
    with pytest.raises(EvidenceError, match="symbolic link|reparse"):
        reject_unsafe_path(tmp_path, "link.txt")


def test_a_file_over_the_limit_is_refused_rather_than_read(tmp_path: Path) -> None:
    oversized = tmp_path / "big.txt"
    oversized.write_bytes(b"x" * (MAX_SOURCE_TEXT_BYTES + 1))
    with pytest.raises(EvidenceError, match="exceeds the"):
        read_bounded_text(oversized, MAX_SOURCE_TEXT_BYTES)


def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError):
        read_bounded_text(tmp_path, MAX_SOURCE_TEXT_BYTES)


def test_a_missing_file_reports_rather_than_raising_oserror(tmp_path: Path) -> None:
    with pytest.raises(EvidenceError, match="Cannot read"):
        read_bounded_text(tmp_path / "absent.txt", MAX_SOURCE_TEXT_BYTES)


def test_duplicate_json_keys_are_refused(tmp_path: Path) -> None:
    document = tmp_path / "doc.json"
    document.write_text('{"schema_version": "embedding-portability.v1", "schema_version": "other"}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="Duplicate JSON key"):
        load_source_store(document, "support-memory")


def test_non_finite_literals_are_refused(tmp_path: Path) -> None:
    document = tmp_path / "doc.json"
    document.write_text('{"schema_version": "embedding-portability.v1", "stores": [NaN]}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="Non-finite"):
        load_source_store(document, "support-memory")


def test_bytes_that_are_not_utf8_are_refused(tmp_path: Path) -> None:
    document = tmp_path / "doc.json"
    document.write_bytes(b'{"schema_version": "\xff\xfe"}')
    with pytest.raises(EvidenceError, match="not valid UTF-8"):
        load_source_store(document, "support-memory")


def test_source_text_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path, texts={"r1": "plain"})
    (tmp_path / "sources" / "r1.txt").write_bytes(b"\xff\xfe\x00")
    store = load_source_store(document, "support-memory")
    with pytest.raises(EvidenceError, match="hashes to|not valid UTF-8"):
        store.read_text(store.retained[0])


def test_a_json_array_at_the_root_is_refused(tmp_path: Path) -> None:
    document = tmp_path / "doc.json"
    document.write_text("[]", encoding="utf-8")
    with pytest.raises(EvidenceError, match="must be a JSON object"):
        load_source_store(document, "support-memory")


def test_the_wrong_schema_version_is_refused(tmp_path: Path) -> None:
    document = write_document(tmp_path, {"schema_version": "embedding-portability.v2", "stores": []})
    with pytest.raises(EvidenceError, match="Expected embedding-portability.v1"):
        load_source_store(document, "support-memory")


def test_an_absent_store_names_what_was_found(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    with pytest.raises(EvidenceError, match="Found: support-memory"):
        load_source_store(document, "other-memory")


def test_a_repeated_store_id_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"].append(payload["stores"][0])
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="appears 2 times"):
        load_source_store(document, "support-memory")


def test_a_repeated_record_id_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    records = payload["stores"][0]["vectors"]["records"]
    records.append(records[0])
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="repeats a record_id"):
        load_source_store(document, "support-memory")


def test_an_unknown_layout_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"][0]["vectors"]["layout"] = "parquet"
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="Unknown vector layout"):
        load_source_store(document, "support-memory")


def test_an_unknown_source_text_status_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"][0]["vectors"]["records"][0]["source_text"] = {"status": "archived"}
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="unknown source_text status"):
        load_source_store(document, "support-memory")


def test_a_store_with_no_records_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"][0]["vectors"]["records"] = []
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="no records"):
        load_source_store(document, "support-memory")


def test_a_record_id_that_is_not_an_identifier_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"][0]["vectors"]["records"][0]["record_id"] = "../escape"
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="not a valid identifier"):
        load_source_store(document, "support-memory")


def test_a_malformed_digest_is_refused(tmp_path: Path) -> None:
    document = build_source_store(tmp_path)
    payload = json.loads(document.read_text(encoding="utf-8"))
    payload["stores"][0]["vectors"]["records"][0]["source_text"]["sha256"] = "ABC"
    write_document(tmp_path, payload)
    with pytest.raises(EvidenceError, match="SHA-256 digest"):
        load_source_store(document, "support-memory")
