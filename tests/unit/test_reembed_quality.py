"""The retrieval-quality comparison: what it measures, and what it refuses to claim.

Every store here is built under ``tmp_path`` from deterministic text. Nothing
embeds through a provider, loads weights or opens a socket, because the
comparison cannot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

import pytest

from private_ai_stack.reembed import (
    DEFAULT_MINIMUM_SAMPLE_SIZE,
    ModelIdentity,
    QualityError,
    QualityThresholds,
    QuerySet,
    attach_quality,
    compare_stores,
    derive_anchor_seed,
    draw_anchors,
    load_query_set,
    load_vector_store,
    re_embed_store,
)
from private_ai_stack.reembed.embedders import build_built_in_embedder
from private_ai_stack.reembed.quality import (
    NEIGHBORHOOD_AGREEMENT,
    NOT_MEASURED,
    OPERATOR,
    QUERY_SET_SCHEMA_VERSION,
    RECORD_ANCHORED,
    RETRIEVAL_RELEVANCE,
    SAMPLED,
    SEEDED_RANDOM,
    TEXT,
    UNJUDGED,
    _average_ranks,
    _spearman,
)
from tests.unit.reembed_support import build_ranked_store

FIXTURES: Final = Path(__file__).resolve().parents[1] / "fixtures/reembed"
OLD_MODEL: Final = ModelIdentity(id="agentforge.hashed-bow.blake2b", revision="1", dimensions=64)
NEW_MODEL: Final = ModelIdentity(id="privateaistack.hashed-bow.sha256-signed", revision="1", dimensions=64)
RECORD_COUNT: Final = 40


def _texts(count: int = RECORD_COUNT) -> list[tuple[str, str]]:
    return [(f"r{index:04d}", f"record {index} about topic {index % 7} and some shared vocabulary here") for index in range(count)]


@pytest.fixture
def stores(tmp_path: Path) -> tuple[Any, Any, Path]:
    """An original store, its rebuild with a different model, and the rebuild's record."""
    source = build_ranked_store(tmp_path / "source")
    outcome = re_embed_store(
        source_document=source,
        source_store_id="old-store",
        target_directory=tmp_path / "rebuilt",
        target_store_id="new-store",
        target_model=NEW_MODEL,
        availability="open_weight_self_hostable",
    )
    before = load_vector_store(source, "old-store")
    after = load_vector_store(outcome.target_document, "new-store")
    return before, after, outcome.record


def _thresholds(**overrides: Any) -> QualityThresholds:
    base: dict[str, Any] = {"top_k": 5, "minimum_sample_size": 10, "minimum_top_k_overlap": 0.1}
    base.update(overrides)
    return QualityThresholds(**base)


# --- what it measures -------------------------------------------------------


def test_a_seeded_record_anchored_comparison_reports_agreement(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="re-embed-quality"),
        thresholds=_thresholds(),
    )

    assert result.status == SAMPLED
    assert result.mode == RECORD_ANCHORED
    assert result.metric_basis == NEIGHBORHOOD_AGREEMENT
    assert result.anchor_selection == SEEDED_RANDOM
    assert result.anchor_seed == "re-embed-quality"
    assert result.query_set_size == 24
    assert result.candidate_pool_size == RECORD_COUNT
    assert 0.0 <= result.metrics["top_k_overlap"] <= 1.0
    assert -1.0 <= result.metrics["rank_correlation"] <= 1.0


def test_record_anchored_agreement_is_never_called_search_quality(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(),
    )

    assert not result.measures_user_search_quality
    assert result.notes is not None
    assert "Not a measurement of search quality" in result.notes
    document = result.to_document()
    assert document["metric_basis"] == NEIGHBORHOOD_AGREEMENT
    # Nothing the module writes may read as a claim about search.
    assert "search quality" not in json.dumps(document).replace("Not a measurement of search quality", "")


def test_a_store_compared_with_itself_agrees_completely(tmp_path: Path) -> None:
    source = build_ranked_store(tmp_path / "source")
    store = load_vector_store(source, "old-store")

    result = compare_stores(
        before=store,
        after=store,
        query_set=draw_anchors(store.record_ids, count=20, seed="s"),
        thresholds=_thresholds(minimum_top_k_overlap=1.0, minimum_rank_correlation=1.0),
    )

    assert result.status == SAMPLED
    assert result.metrics["top_k_overlap"] == pytest.approx(1.0)
    assert result.metrics["rank_correlation"] == pytest.approx(1.0)


def test_an_anchor_is_never_its_own_neighbour(stores: tuple[Any, Any, Path]) -> None:
    # An anchor ranks first against itself in both stores, so leaving it in
    # would add one guaranteed agreement to every single query.
    before, after, _ = stores
    single = QuerySet(mode=RECORD_ANCHORED, selection=SEEDED_RANDOM, seed="s", anchors=("r0000",))

    result = compare_stores(
        before=before,
        after=after,
        query_set=single,
        thresholds=_thresholds(minimum_sample_size=1),
    )

    # With the anchor excluded the pool is one smaller than the store.
    assert result.query_set_size == 1
    assert result.candidate_pool_size == RECORD_COUNT


# --- who chose the queries --------------------------------------------------


def test_an_operator_query_set_is_reported_but_marked_operator_chosen(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    queries = load_query_set(FIXTURES / "golden-query-set.json")

    result = compare_stores(before=before, after=after, query_set=queries, thresholds=_thresholds(minimum_sample_size=5))

    assert result.anchor_selection == OPERATOR
    assert result.anchor_seed is None
    assert result.status in (SAMPLED, "below_threshold")
    assert result.metrics["top_k_overlap"] >= 0.0
    assert "anchor_seed" not in result.to_document()


def test_a_drawn_anchor_set_is_reproducible_from_its_seed(stores: tuple[Any, Any, Path]) -> None:
    before, _, _ = stores

    first = draw_anchors(before.record_ids, count=12, seed="one")
    again = draw_anchors(before.record_ids, count=12, seed="one")
    other = draw_anchors(before.record_ids, count=12, seed="two")

    assert first.anchors == again.anchors
    assert first.anchors != other.anchors
    assert first.selection == SEEDED_RANDOM


def test_the_draw_is_not_simply_the_first_records_by_identifier(stores: tuple[Any, Any, Path]) -> None:
    before, _, _ = stores

    drawn = draw_anchors(before.record_ids, count=12, seed="re-embed-quality")

    assert drawn.anchors != tuple(sorted(before.record_ids)[:12])


# --- thresholds -------------------------------------------------------------


def test_a_comparison_with_no_metric_threshold_is_unjudged_not_discarded(
    stores: tuple[Any, Any, Path],
) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=QualityThresholds(top_k=5, minimum_sample_size=10),
    )

    assert result.status == UNJUDGED
    assert not result.judged
    assert not result.passed
    # The measurement is real; only the verdict is absent.
    assert 0.0 <= result.metrics["top_k_overlap"] <= 1.0
    assert result.sufficient_evidence
    assert result.query_set_size == 24


def test_an_unjudged_result_names_the_floors_that_would_give_a_verdict(
    stores: tuple[Any, Any, Path],
) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=QualityThresholds(top_k=5, minimum_sample_size=10),
    )

    assert result.notes is not None
    assert "minimum_top_k_overlap" in result.notes
    assert "minimum_rank_correlation" in result.notes
    # Anchors have no expected records, so the drop caps are not an option here.
    assert "anchors do not have" in result.notes
    assert len(result.notes) <= 512


def test_an_unjudged_result_carries_only_the_sample_size_check(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=QualityThresholds(top_k=5, minimum_sample_size=10),
    )

    assert [check.name for check in result.checks] == ["minimum_sample_size"]
    assert all(check.passed for check in result.checks)
    assert set(result.thresholds) == {"minimum_sample_size"}


def test_an_unjudged_result_can_never_move_a_score(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    seeded = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=QualityThresholds(top_k=5, minimum_sample_size=10),
    )

    # Even a seeded draw, which is the only kind that can move a score, cannot
    # move one without a verdict to move it on.
    assert seeded.anchor_selection == SEEDED_RANDOM
    assert not seeded.passed
    assert not seeded.judged


def test_text_mode_without_expected_records_is_told_about_them(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    queries = QuerySet(mode=TEXT, selection=OPERATOR, seed=None, queries=(("shared vocabulary", ()),))

    result = compare_stores(
        before=before,
        after=after,
        query_set=queries,
        thresholds=QualityThresholds(top_k=5, minimum_sample_size=1),
    )

    assert result.status == UNJUDGED
    assert result.notes is not None
    assert "Expected records would also bring the drop caps into play" in result.notes


def test_the_declared_thresholds_travel_with_the_verdict(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(minimum_rank_correlation=0.0),
    )

    assert set(result.thresholds) == {check.name for check in result.checks}
    assert result.thresholds["minimum_top_k_overlap"] == 0.1
    summary = result.threshold_summary()
    assert summary is not None and "minimum_rank_correlation" in summary


def test_an_unreachable_floor_produces_below_threshold_not_a_failure(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(minimum_top_k_overlap=1.0),
    )

    assert result.status == "below_threshold"
    assert not result.passed
    assert any(not check.passed for check in result.checks)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("top_k", 0),
        ("top_k", 100_000),
        ("minimum_sample_size", 0),
        ("minimum_top_k_overlap", 1.5),
        ("minimum_rank_correlation", -2.0),
        ("maximum_mrr_drop", 2.0),
    ),
)
def test_an_out_of_range_threshold_is_refused(field: str, value: float) -> None:
    with pytest.raises(QualityError):
        QualityThresholds(**{field: value})


def test_the_minimum_sample_size_matches_the_benchmarking_default() -> None:
    # The two projects take no dependency on each other, so the number is
    # restated. A drift between them would make one tool's calibration useless
    # in the other.
    assert DEFAULT_MINIMUM_SAMPLE_SIZE == 20
    assert QualityThresholds().minimum_sample_size == 20


def test_no_agreement_floor_is_shipped() -> None:
    # A floor this package invented would be a bar nobody set.
    defaults = QualityThresholds()
    assert defaults.minimum_top_k_overlap is None
    assert defaults.minimum_rank_correlation is None


# --- empty, small and skipped query sets ------------------------------------


def test_an_empty_query_set_is_not_measured_rather_than_passing(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=0, seed="s"),
        thresholds=_thresholds(),
    )

    assert result.status == NOT_MEASURED
    assert result.query_set_size == 0
    assert not result.sufficient_evidence
    assert not result.passed
    assert result.notes is not None and "not a passing result" in result.notes
    # not_measured carries no metrics, so no number can be read out of it.
    assert result.to_document().get("metrics") is None


def test_a_query_set_below_the_minimum_is_not_measured(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=4, seed="s"),
        thresholds=_thresholds(minimum_sample_size=20),
    )

    assert result.status == NOT_MEASURED
    assert result.query_set_size == 4
    assert not result.sufficient_evidence
    assert result.notes is not None and "under the declared minimum" in result.notes


def test_anchors_missing_from_a_store_are_skipped_not_fatal(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    queries = QuerySet(
        mode=RECORD_ANCHORED,
        selection=OPERATOR,
        seed=None,
        anchors=("r0000", "r0001", "nosuchrecord", "r0002", "alsomissing"),
    )

    result = compare_stores(before=before, after=after, query_set=queries, thresholds=_thresholds(minimum_sample_size=3))

    assert result.query_set_size == 3
    assert result.queries_skipped == 2


def test_two_stores_with_no_records_in_common_are_refused(tmp_path: Path) -> None:
    before = load_vector_store(build_ranked_store(tmp_path / "a", count=4), "old-store")
    after = load_vector_store(build_ranked_store(tmp_path / "b", count=4, model_id=NEW_MODEL.id), "old-store")
    after.vectors.clear()
    after.vectors["elsewhere"] = next(iter(before.vectors.values()))

    with pytest.raises(QualityError, match="share no record identifiers"):
        compare_stores(
            before=before,
            after=after,
            query_set=draw_anchors(before.record_ids, count=2, seed="s"),
            thresholds=_thresholds(minimum_sample_size=1),
        )


# --- ties -------------------------------------------------------------------


def test_equal_similarities_are_ordered_by_identifier_not_by_chance(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(),
    )

    assert result.to_document()["tie_break"] == "ascending record_id among equal similarities"
    assert result.boundary_ties >= 0


def test_ranks_average_across_a_tie() -> None:
    # Breaking a tie by position would make the correlation depend on the
    # identifier order rather than on the vectors.
    assert _average_ranks([1.0, 1.0, 2.0]) == [1.5, 1.5, 3.0]
    assert _average_ranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]


def test_a_correlation_with_no_ordering_at_all_is_undefined_not_zero() -> None:
    # Every candidate tied, so there is no ordering to agree or disagree with.
    assert _spearman([1.0, 1.0, 1.0], [0.3, 0.1, 0.2]) is None
    assert _spearman([1.0], [1.0]) is None
    assert _spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_a_boundary_tie_is_counted_rather_than_hidden(tmp_path: Path) -> None:
    # Identical texts give identical vectors, so every similarity ties and the
    # top-k boundary is decided by the tie break rather than by the vectors.
    source = tmp_path / "flat"
    (source / "sources").mkdir(parents=True)
    (source / "vectors").mkdir(parents=True)
    embedder = build_built_in_embedder(OLD_MODEL)
    entries, lines = [], []
    for index in range(12):
        text = "identical text in every record"
        relative = f"sources/record-{index:04d}.txt"
        (source / relative).write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()
        entries.append({"record_id": f"r{index:04d}", "source_text": {"status": "retained", "path": relative, "sha256": digest}})
        lines.append(json.dumps({"record_id": f"r{index:04d}", "source_text_sha256": digest, "values": embedder.embed(text)}))
    shard = ("\n".join(lines) + "\n").encode()
    (source / "vectors/shard-0000.jsonl").write_bytes(shard)
    (source / "embedding-portability.json").write_text(
        json.dumps(
            {
                "schema_version": "embedding-portability.v1",
                "stores": [
                    {
                        "store_id": "flat",
                        "embedding_model": {
                            "id": OLD_MODEL.id,
                            "revision": "1",
                            "dimensions": 64,
                            "availability": "open_weight_self_hostable",
                            "availability_verification": "attested",
                        },
                        "vectors": {
                            "layout": "sharded_jsonl",
                            "shards": [
                                {
                                    "path": "vectors/shard-0000.jsonl",
                                    "sha256": hashlib.sha256(shard).hexdigest(),
                                    "record_count": 12,
                                }
                            ],
                            "source_text": entries,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = load_vector_store(source / "embedding-portability.json", "flat")

    result = compare_stores(
        before=store,
        after=store,
        query_set=draw_anchors(store.record_ids, count=12, seed="s"),
        thresholds=_thresholds(minimum_sample_size=1, minimum_top_k_overlap=0.0),
    )

    assert result.boundary_ties == 12
    assert result.correlation_undefined
    assert "rank_correlation" not in result.metrics


def test_a_top_k_larger_than_the_pool_is_capped_and_said_so(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=20, seed="s"),
        thresholds=_thresholds(top_k=100),
        candidate_pool=8,
    )

    assert result.top_k_capped
    assert result.candidate_pool_size <= RECORD_COUNT
    assert result.top_k == 100


# --- query sets on disk -----------------------------------------------------


def test_the_shipped_record_anchored_fixture_loads() -> None:
    queries = load_query_set(FIXTURES / "golden-query-set.json")

    assert queries.mode == RECORD_ANCHORED
    assert queries.selection == OPERATOR
    assert queries.seed is None
    assert queries.anchors == ("r0000", "r0003", "r0007", "r0011", "r0019")


def test_the_shipped_text_fixture_loads_with_its_expected_records() -> None:
    queries = load_query_set(FIXTURES / "golden-query-set-text.json")

    assert queries.mode == TEXT
    assert queries.has_expected_records
    assert queries.queries[0][1] == ("r0003", "r0010")
    assert queries.queries[2][1] == ()


def test_a_query_set_from_a_file_is_always_operator_selected(tmp_path: Path) -> None:
    # The file is what the operator chose to hand over, whatever it says about
    # itself, so a set cannot declare its way into moving a score.
    path = tmp_path / "set.json"
    path.write_text(
        json.dumps({"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": RECORD_ANCHORED, "anchors": ["r0000"]}),
        encoding="utf-8",
    )

    assert load_query_set(path).selection == OPERATOR


@pytest.mark.parametrize(
    ("document", "message"),
    (
        ({"schema_version": "golden-query-set.v2", "mode": RECORD_ANCHORED, "anchors": []}, "golden-query-set.v1"),
        ({"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": "vibes"}, "Unknown query set mode"),
        (
            {"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": RECORD_ANCHORED, "anchors": ["a", "a"]},
            "repeats a record identifier",
        ),
        (
            {"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": RECORD_ANCHORED, "anchors": ["a"], "queries": []},
            "carries anchors, not queries",
        ),
        (
            {"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": TEXT, "queries": [], "anchors": []},
            "carries queries, not anchors",
        ),
        ({"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": TEXT, "queries": [{"text": ""}]}, "needs text"),
        ({"schema_version": QUERY_SET_SCHEMA_VERSION, "mode": TEXT, "queries": ["a query"]}, "must be a JSON object"),
    ),
)
def test_a_malformed_query_set_is_refused(tmp_path: Path, document: dict[str, Any], message: str) -> None:
    path = tmp_path / "set.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(QualityError, match=message):
        load_query_set(path)


# --- text mode --------------------------------------------------------------


def test_text_queries_are_embedded_by_each_store_own_model(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    queries = load_query_set(FIXTURES / "golden-query-set-text.json")

    result = compare_stores(before=before, after=after, query_set=queries, thresholds=_thresholds(minimum_sample_size=3))

    assert result.mode == TEXT
    assert result.metric_basis == RETRIEVAL_RELEVANCE
    assert result.measures_user_search_quality
    assert "hit_rate" in result.metrics
    assert "hit_rate_before" in result.metrics
    assert "mrr" in result.metrics


def test_text_mode_without_expected_records_is_agreement_again(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    queries = QuerySet(mode=TEXT, selection=OPERATOR, seed=None, queries=(("shared vocabulary", ()),))

    result = compare_stores(before=before, after=after, query_set=queries, thresholds=_thresholds(minimum_sample_size=1))

    assert result.metric_basis == NEIGHBORHOOD_AGREEMENT
    assert not result.measures_user_search_quality
    assert "hit_rate" not in result.metrics


def test_text_mode_against_a_model_this_package_cannot_run_is_refused(tmp_path: Path) -> None:
    source = build_ranked_store(tmp_path / "source")
    store = load_vector_store(source, "old-store")
    foreign = type(store)(
        store_id="foreign",
        model=ModelIdentity(id="some-vendor-embed", revision="1", dimensions=64),
        vectors=store.vectors,
    )
    queries = QuerySet(mode=TEXT, selection=OPERATOR, seed=None, queries=(("anything", ()),))

    with pytest.raises(QualityError, match="does not implement"):
        compare_stores(before=foreign, after=store, query_set=queries, thresholds=_thresholds(minimum_sample_size=1))


# --- reading stores ---------------------------------------------------------


def test_a_shard_that_drifted_from_its_digest_is_refused(tmp_path: Path) -> None:
    source = build_ranked_store(tmp_path / "source")
    shard = tmp_path / "source/vectors/shard-0000.jsonl"
    shard.write_bytes(shard.read_bytes().replace(b"r0000", b"r9999"))

    with pytest.raises(QualityError, match="differs from the digest"):
        load_vector_store(source, "old-store")


def test_a_single_file_layout_is_read_too(tmp_path: Path) -> None:
    embedder = build_built_in_embedder(OLD_MODEL)
    root = tmp_path / "single"
    (root / "vectors").mkdir(parents=True)
    (root / "sources").mkdir(parents=True)
    records = []
    for index, (record_id, text) in enumerate(_texts(6)):
        values = {"values": embedder.embed(text)}
        payload = json.dumps(values).encode("utf-8")
        (root / f"vectors/v-{index:04d}.json").write_bytes(payload)
        (root / f"sources/record-{index:04d}.txt").write_text(text, encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()
        records.append(
            {
                "record_id": record_id,
                "vector": {"path": f"vectors/v-{index:04d}.json", "sha256": hashlib.sha256(payload).hexdigest()},
                "source_text_sha256_in_vector_record": digest,
                "source_text": {"status": "retained", "path": f"sources/record-{index:04d}.txt", "sha256": digest},
            }
        )
    path = root / "embedding-portability.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "embedding-portability.v1",
                "stores": [
                    {
                        "store_id": "single",
                        "embedding_model": {
                            "id": OLD_MODEL.id,
                            "revision": "1",
                            "dimensions": 64,
                            "availability": "open_weight_self_hostable",
                            "availability_verification": "attested",
                        },
                        "vectors": {"layout": "single_file", "records": records},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    store = load_vector_store(path, "single")

    assert len(store.vectors) == 6
    assert len(store.vectors["r0000"]) == 64


def test_an_unknown_store_identifier_is_refused(tmp_path: Path) -> None:
    source = build_ranked_store(tmp_path / "source")

    with pytest.raises(QualityError, match="Expected exactly one store"):
        load_vector_store(source, "nosuchstore")


def test_a_document_of_the_wrong_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"schema_version": "something-else.v1", "stores": []}), encoding="utf-8")

    with pytest.raises(QualityError, match="embedding-portability.v1"):
        load_vector_store(path, "any")


# --- writing the record -----------------------------------------------------


def test_the_quality_block_is_written_into_the_re_embed_record(stores: tuple[Any, Any, Path]) -> None:
    before, after, record_path = stores
    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="re-embed-quality"),
        thresholds=_thresholds(),
    )

    document = attach_quality(record_path, result)

    assert document["schema_version"] == "re-embed-record.v1"
    assert document["quality"]["status"] == SAMPLED
    assert document["quality"]["anchor_selection"] == SEEDED_RANDOM
    assert document["quality"]["anchor_seed"] == "re-embed-quality"
    assert document["quality"]["metric_basis"] == NEIGHBORHOOD_AGREEMENT
    assert document["quality"]["candidate_pool_size"] == RECORD_COUNT
    assert json.loads(record_path.read_text(encoding="utf-8")) == document


def test_a_second_comparison_replaces_the_first_rather_than_merging(stores: tuple[Any, Any, Path]) -> None:
    before, after, record_path = stores
    first = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(),
    )
    attach_quality(record_path, first)
    second = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=0, seed="s"),
        thresholds=_thresholds(),
    )

    document = attach_quality(record_path, second)

    # Two comparisons are two answers, not one merged one.
    assert document["quality"]["status"] == NOT_MEASURED
    assert "metrics" not in document["quality"]


def test_writing_into_something_that_is_not_a_record_is_refused(tmp_path: Path, stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(),
    )
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema_version": "exit-report.v1"}), encoding="utf-8")

    with pytest.raises(QualityError, match="re-embed-record.v1"):
        attach_quality(other, result)


# --- bounds -----------------------------------------------------------------


def test_a_comparison_too_large_to_run_is_refused_rather_than_attempted() -> None:
    # The guard has to fire before any ranking starts, so this builds the
    # shape of a store too wide to rank rather than the bytes of one: the
    # point is that nothing is attempted, not that it is slow.
    from array import array

    from private_ai_stack.reembed.quality import VectorStore

    wide = ModelIdentity(id=OLD_MODEL.id, revision="1", dimensions=8192)
    vectors = {f"r{index:04d}": array("d", [1.0]) for index in range(1100)}
    before = VectorStore(store_id="wide", model=wide, vectors=vectors)
    after = VectorStore(store_id="wide", model=wide, vectors=dict(vectors))

    with pytest.raises(QualityError, match="over the"):
        compare_stores(
            before=before,
            after=after,
            query_set=draw_anchors(before.record_ids, count=1000, seed="s"),
            thresholds=_thresholds(),
            candidate_pool=20_000,
        )


@pytest.mark.parametrize("pool", (0, 20_001))
def test_an_out_of_range_candidate_pool_is_refused(stores: tuple[Any, Any, Path], pool: int) -> None:
    before, after, _ = stores

    with pytest.raises(QualityError, match="candidate_pool"):
        compare_stores(
            before=before,
            after=after,
            query_set=draw_anchors(before.record_ids, count=4, seed="s"),
            thresholds=_thresholds(),
            candidate_pool=pool,
        )


@pytest.mark.parametrize(("count", "seed"), ((-1, "s"), (1001, "s"), (4, ""), (4, "x" * 129)))
def test_an_out_of_range_draw_is_refused(stores: tuple[Any, Any, Path], count: int, seed: str) -> None:
    before, _, _ = stores

    with pytest.raises(QualityError):
        draw_anchors(before.record_ids, count=count, seed=seed)


def test_the_comparison_opens_no_socket(stores: tuple[Any, Any, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("The quality comparison must not open a socket.")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    before, after, _ = stores

    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(),
    )

    assert result.status in (SAMPLED, "below_threshold")


# --- the format itself ------------------------------------------------------

QUERY_SET_SCHEMA: Final = Path(__file__).resolve().parents[2] / "private_ai_stack/reembed/schemas/golden-query-set.schema.json"


@pytest.mark.parametrize("name", ("golden-query-set.json", "golden-query-set-text.json"))
def test_the_shipped_query_set_fixtures_match_the_schema_contract(name: str) -> None:
    schema = json.loads(QUERY_SET_SCHEMA.read_text(encoding="utf-8"))
    document = json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"]["const"] == QUERY_SET_SCHEMA_VERSION
    assert document["schema_version"] == QUERY_SET_SCHEMA_VERSION
    assert set(schema["required"]) <= set(document)
    assert set(document) <= set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_the_two_fixtures_cover_both_modes() -> None:
    modes = {
        json.loads((FIXTURES / name).read_text(encoding="utf-8"))["mode"]
        for name in ("golden-query-set.json", "golden-query-set-text.json")
    }

    assert modes == {RECORD_ANCHORED, TEXT}


def test_the_schema_keeps_the_two_modes_from_mixing() -> None:
    schema = json.loads(QUERY_SET_SCHEMA.read_text(encoding="utf-8"))
    rules = {rule["if"]["properties"]["mode"]["const"]: rule["then"] for rule in schema["allOf"]}

    assert rules[RECORD_ANCHORED]["required"] == ["anchors"]
    assert rules[RECORD_ANCHORED]["not"]["required"] == ["queries"]
    assert rules[TEXT]["required"] == ["queries"]
    assert rules[TEXT]["not"]["required"] == ["anchors"]


def test_the_quality_block_names_every_field_the_result_carries(stores: tuple[Any, Any, Path]) -> None:
    # The suite's re-embed-record.v1 quality block is additionalProperties:
    # false, so a field added here without a matching field there would make
    # every record this writes unreadable.
    before, after, _ = stores
    result = compare_stores(
        before=before,
        after=after,
        query_set=draw_anchors(before.record_ids, count=24, seed="s"),
        thresholds=_thresholds(minimum_rank_correlation=0.0),
    )

    assert set(result.to_document()) == {
        "status",
        "query_set_size",
        "top_k",
        "sufficient_evidence",
        "mode",
        "metric_basis",
        "anchor_selection",
        "anchor_seed",
        "similarity",
        "candidate_pool_size",
        "tie_break",
        "boundary_ties",
        "queries_skipped",
        "correlation_undefined",
        "top_k_capped",
        "metrics",
        "checks",
        "thresholds",
        "notes",
    }


# --- 3b-6: seeds are derived, never accepted ------------------------------


def test_the_anchor_seed_is_derived_from_both_store_digests() -> None:
    first = derive_anchor_seed("a" * 64, "b" * 64)

    assert first == derive_anchor_seed("a" * 64, "b" * 64)
    assert len(first) == 64
    # Not symmetric: source and target are not interchangeable.
    assert first != derive_anchor_seed("b" * 64, "a" * 64)
    assert first != derive_anchor_seed("a" * 64, "c" * 64)


def test_a_derived_seed_draw_reports_as_seeded_random(stores: tuple[Any, Any, Path]) -> None:
    before, _, _ = stores

    drawn = draw_anchors(before.record_ids, count=4, seed=derive_anchor_seed("a" * 64, "b" * 64))

    assert drawn.selection == SEEDED_RANDOM
    assert drawn.seed is not None


def test_a_chosen_seed_reports_as_operator_even_though_the_draw_is_mechanical(
    stores: tuple[Any, Any, Path],
) -> None:
    # An operator who can pick the seed can keep picking until the numbers
    # pass, so the provenance of the seed decides how the set is labelled.
    before, _, _ = stores

    drawn = draw_anchors(before.record_ids, count=4, seed="operator-picked", derived=False)

    assert drawn.selection == OPERATOR
    assert drawn.seed is None


def test_a_chosen_seed_still_draws_the_same_anchors(stores: tuple[Any, Any, Path]) -> None:
    # Only the label changes, not the arithmetic: the result is still
    # reproducible, it just cannot be trusted to be unchosen.
    before, _, _ = stores

    derived = draw_anchors(before.record_ids, count=4, seed="same-seed", derived=True)
    chosen = draw_anchors(before.record_ids, count=4, seed="same-seed", derived=False)

    assert derived.anchors == chosen.anchors
    assert derived.selection != chosen.selection


def test_a_chosen_seed_records_no_seed_in_the_result(stores: tuple[Any, Any, Path]) -> None:
    before, after, _ = stores
    chosen = draw_anchors(before.record_ids, count=24, seed="operator-picked", derived=False)

    result = compare_stores(before=before, after=after, query_set=chosen, thresholds=_thresholds())

    document = result.to_document()
    assert document["anchor_selection"] == OPERATOR
    assert "anchor_seed" not in document
