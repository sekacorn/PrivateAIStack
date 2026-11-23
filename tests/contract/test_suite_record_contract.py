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

import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.reembed_record_fixtures import VARIANTS, build_all

suite_reader = pytest.importorskip(
    "ai_infrastructure_suite.re_embed_record",
    reason="ai-infrastructure-suite is not installed; the contract job installs it from the suite's main branch.",
)
jsonschema = pytest.importorskip("jsonschema", reason="jsonschema is not installed; the contract job installs it.")


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
