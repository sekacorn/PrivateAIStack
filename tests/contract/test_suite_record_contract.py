"""Every record this package emits must be readable by the AI Infrastructure Suite.

The two projects take no code dependency on each other; the contract between
them is the ``re-embed-record.v1`` document. Nothing in the normal test suite
checks that contract, which is how PrivateAIStack came to emit twenty-one
egress fields against a reader that knew ten. Every record it produced was
rejected outright, and no test noticed.

This module closes that gap from this side. It is deliberately **not** part of
the offline unit suite: it needs ``ai-infrastructure-suite`` installed, which
CI does from the suite's main branch. Without it every test here skips, so the
offline matrix stays offline and stays green.

The test validates through **both** halves of the contract, because they fail
differently. The reader enforces cross-field consistency a JSON Schema cannot
express; the schema enforces the field inventory with
``additionalProperties: false``. A record has to satisfy both.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Final

import pytest

from tests.unit.reembed_record_fixtures import VARIANTS, build_all

REQUIRE_ENV: Final = "PRIVATEAISTACK_REQUIRE_SUITE_CONTRACT"
"""Set by CI. Turns a missing suite from a skip into a failure.

A skip is the right answer on a developer's offline machine, where the suite is
not installed and the unit matrix must stay offline. It is the wrong answer in
the job whose entire purpose is to check the contract: a silent skip there
would report success for a check that never ran, and this job is about to
become a required status check. So CI sets this and the import becomes hard.
"""


def _import_or_fail(module: str, package: str) -> Any:
    required = os.environ.get(REQUIRE_ENV) == "1"
    try:
        return importlib.import_module(module)
    except ImportError:
        if required:
            pytest.fail(
                f"{package} is not installed and {REQUIRE_ENV}=1, so this contract check "
                "cannot run. The job that sets it must install the suite before running these "
                "tests; a skip here would report success for a check that never happened."
            )
        # allow_module_level: this runs during collection, not inside a test.
        pytest.skip(
            f"{package} is not installed; set {REQUIRE_ENV}=1 to make this a failure instead.",
            allow_module_level=True,
        )


suite_reader = _import_or_fail("ai_infrastructure_suite.re_embed_record", "ai-infrastructure-suite")
jsonschema = _import_or_fail("jsonschema", "jsonschema")


def _suite_schema() -> dict[str, Any]:
    from importlib.resources import files

    raw = files("ai_infrastructure_suite").joinpath("data/re-embed-record.schema.json").read_text(encoding="utf-8")
    return json.loads(raw)


@pytest.fixture(scope="module")
def emitted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """Every record variant, produced by running the real pipeline."""
    return build_all(tmp_path_factory.mktemp("contract"))


def test_the_installed_suite_is_reported(emitted: dict[str, dict[str, Any]]) -> None:
    # Printed into the CI log so a failure says which suite it was run against.
    import ai_infrastructure_suite

    print(f"validated against ai-infrastructure-suite {ai_infrastructure_suite.__version__}")
    assert emitted


def test_every_variant_is_covered(emitted: dict[str, dict[str, Any]]) -> None:
    # A new record shape cannot be added without this job validating it.
    assert set(emitted) == set(VARIANTS)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_suite_reader_accepts_the_record(variant: str, emitted: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    path = tmp_path / f"{variant}.json"
    path.write_text(json.dumps(emitted[variant]), encoding="utf-8")

    record = suite_reader.load_re_embed_record(path)

    assert record.egress is not None


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_suite_schema_accepts_the_record(variant: str, emitted: dict[str, dict[str, Any]]) -> None:
    jsonschema.validate(emitted[variant], _suite_schema())


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_unknown_egress_field_is_rejected(variant: str, emitted: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    """The check has teeth: proving it fails is what makes a pass mean anything.

    Both halves must refuse an unknown field. If either stopped doing so, this
    job would go on passing while the contract quietly rotted.
    """
    document = json.loads(json.dumps(emitted[variant]))
    document["egress"]["speculative_new_field"] = "whatever"
    path = tmp_path / f"{variant}-unknown.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(suite_reader.ReEmbedRecordError, match="Unexpected fields in egress"):
        suite_reader.load_re_embed_record(path)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _suite_schema())


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_variant_can_claim_delivery(variant: str, emitted: dict[str, dict[str, Any]], tmp_path: Path) -> None:
    path = tmp_path / f"{variant}.json"
    path.write_text(json.dumps(emitted[variant]), encoding="utf-8")

    egress = suite_reader.load_re_embed_record(path).egress

    assert not egress.delivery_established
    if egress.used_external_executor:
        assert egress.delivery_verification == "unverified"
        assert egress.transmission_basis == "handed_to_executor"
        assert egress.transmission_attempts_may_exceed_count
    else:
        assert egress.transmission_basis == "no_transmission"
        assert egress.stayed_local


def test_the_writer_always_states_a_transmission_basis(emitted: dict[str, dict[str, Any]]) -> None:
    # The suite deliberately does not require this, so that records written
    # before the field stay valid. This is the writer's half of that bargain,
    # and it is checked here rather than assumed.
    for variant, document in emitted.items():
        assert document["egress"].get("transmission_basis"), variant


def test_every_emitted_egress_field_is_known_to_the_suite(emitted: dict[str, dict[str, Any]]) -> None:
    """Report the whole field-inventory gap at once, not one error per run.

    A parametrised failure names a single offending field; this names all of
    them, which is what someone reconciling the two repositories needs.
    """
    allowed = set(_suite_schema()["$defs"]["egress"]["properties"])
    unknown: dict[str, list[str]] = {}
    for variant, document in emitted.items():
        extra = sorted(set(document["egress"]) - allowed)
        if extra:
            unknown[variant] = extra

    assert not unknown, f"egress fields the installed suite does not know: {unknown}"


# --- 3b-6: the two implementations must not drift -------------------------

AGREEMENT_TEXTS: Final = (
    "Escalate to the on-call engineer when the queue depth exceeds 40.",
    "Refunds over 500 need a supervisor approval and an audit entry.",
    "repeated repeated repeated repeated",
    "MiXeD CaSe Tokens 123 456",
    "ünïcödé tëxt wíth áccents",
    "a",
    "   ",
    "!!! ??? ...",
    "tab\tseparated\twords",
    "trailing whitespace   ",
)
AGREEMENT_DIMENSIONS: Final = (1, 8, 16, 64, 384)


def test_the_suite_reimplements_the_same_arithmetic() -> None:
    """The cross-repo agreement test the 3b-6 design turns on.

    The suite reproduces a rebuild with its *own* copy of these two schemes,
    because using this package's embedders to verify this package's claims
    would not be independent verification. The price of a third
    implementation is drift, and this is where drift surfaces: not in a shared
    fixture two repositories have to keep in step by hand, but in the two
    implementations being run against each other.

    If this fails, every record this package writes silently stops being able
    to move a lock-in score, because the suite's recomputation will not match.
    """
    suite_reproduction = _import_or_fail("ai_infrastructure_suite.reproduction", "ai-infrastructure-suite")
    from private_ai_stack.reembed.embedders import BUILT_IN_MODEL_IDS, ModelIdentity, build_built_in_embedder

    compared = 0
    for model_id in sorted(BUILT_IN_MODEL_IDS):
        for dimensions in AGREEMENT_DIMENSIONS:
            embedder = build_built_in_embedder(ModelIdentity(id=model_id, revision="1", dimensions=dimensions))
            assert suite_reproduction.is_reproducible(model_id, "1", dimensions), (
                f"The suite does not recognise {model_id}@1 at {dimensions} dimensions, so it could never reproduce a rebuild onto it."
            )
            for text in AGREEMENT_TEXTS:
                ours = embedder.embed(text)
                theirs = suite_reproduction.embed(text, model_id=model_id, revision="1", dimensions=dimensions)
                assert suite_reproduction.vectors_match(theirs, ours), (
                    f"Drift: {model_id}@1 at {dimensions} dimensions disagrees on {text!r}."
                )
                compared += 1
    assert compared == len(BUILT_IN_MODEL_IDS) * len(AGREEMENT_DIMENSIONS) * len(AGREEMENT_TEXTS)


def test_both_sides_derive_the_same_anchor_seed() -> None:
    """A seed the two sides disagree about is a score that never moves.

    The suite treats any anchor seed other than the one it derives as an
    operator-chosen query set. If these formulas diverge, every record this
    package writes is read as hand-picked and moves nothing -- silently, with
    no error anywhere.
    """
    suite_scan = _import_or_fail("ai_infrastructure_suite.re_embed_scan", "ai-infrastructure-suite")
    from private_ai_stack.reembed import derive_anchor_seed

    for source, target in (("a" * 64, "b" * 64), ("0" * 64, "f" * 64), ("1" * 64, "1" * 64)):
        assert derive_anchor_seed(source, target) == suite_scan.derive_anchor_seed(source, target)
    # And it is not accidentally symmetric, which would make source and target
    # interchangeable.
    assert derive_anchor_seed("a" * 64, "b" * 64) != derive_anchor_seed("b" * 64, "a" * 64)


def test_a_record_this_package_writes_earns_the_verified_level(tmp_path: Path) -> None:
    """The whole contract, end to end, in the one case that is supposed to work.

    A vendor-locked store rebuilt onto a self-hostable built-in scheme, with a
    derived anchor seed, must come out of the suite as a reproduced rebuild
    that moves 50 to 25. Every other test here checks a piece; this checks
    that the pieces still add up.
    """
    suite_scan = _import_or_fail("ai_infrastructure_suite.re_embed_scan", "ai-infrastructure-suite")
    import hashlib

    from private_ai_stack.reembed import derive_anchor_seed
    from tests.unit.reembed_record_fixtures import (
        MODEL,
        build_hosted_record,  # noqa: F401  (import check)
        build_local_record,
    )

    record_path = build_local_record(tmp_path)
    document = json.loads(record_path.read_text(encoding="utf-8"))
    source_sha = document["source_store"]["store_evidence_sha256"]
    target_sha = document["target_store"]["store_evidence_sha256"]
    document["quality"] = {
        "status": "sampled",
        "query_set_size": 24,
        "top_k": 5,
        "sufficient_evidence": True,
        "mode": "record_anchored",
        "metric_basis": "neighborhood_agreement",
        "anchor_selection": "seeded_random",
        "anchor_seed": derive_anchor_seed(source_sha, target_sha),
        "similarity": "cosine",
        "thresholds": {"minimum_top_k_overlap": 0.7},
        "metrics": {"top_k_overlap": 0.81},
        "checks": [{"name": "minimum_top_k_overlap", "passed": True, "actual": 0.81, "threshold": 0.7}],
        "notes": "Neighborhood agreement between the two stores.",
    }
    record_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    assessment = suite_scan.scan_re_embed_record(
        record_path,
        declared_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        source_store_sha256=source_sha,
        source_store_ids=frozenset({"support-memory"}),
    )

    assert assessment.reproduced, assessment.reason
    assert assessment.anchor_selection == "seeded_random"
    assert not assessment.seed_was_ground
    assert assessment.label == "verified (re-embedded, quality sampled)"
    assert MODEL.id.startswith("privateaistack.")
