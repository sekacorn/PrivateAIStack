"""Golden vectors: an embedder's output may not change under an unchanged revision.

A model identity is a promise about arithmetic. If
``privateaistack.hashed-bow.sha256-signed`` revision ``1`` produces one vector
today and a different one after a refactor, then every store, record and
digest produced by the old code describes a model that no longer exists, and
nothing downstream can tell: the identity in the evidence still reads the same.

So the output is pinned. ``tests/fixtures/reembed/golden-vectors.json`` holds
the exact vectors each built-in identity produces for a fixed set of texts, and
these tests fail if any of them moves. The fixture is a test asset rather than
an evidence format, so no schema ships for it; its ``schema_version`` is there
so a future shape change is legible.

Changing an embedder's output is allowed. Editing this fixture to match the new
output is not: the change requires a **new revision** in the model identity,
with the old revision either kept or removed deliberately. See
``docs/re-embedding.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from private_ai_stack.reembed.embedders import (
    BUILT_IN_MODEL_IDS,
    ModelIdentity,
    build_built_in_embedder,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reembed" / "golden-vectors.json"
SCHEMA_VERSION = "embedder-golden-vectors.v1"


def _fixture() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return document


def _identities() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = _fixture()["identities"]
    return entries


def _case_id(entry: dict[str, Any]) -> str:
    return f"{entry['id']}@{entry['revision']}x{entry['dimensions']}"


def test_the_fixture_declares_its_version() -> None:
    assert _fixture()["schema_version"] == SCHEMA_VERSION


def test_every_built_in_model_id_has_golden_vectors() -> None:
    """A new built-in scheme without pinned output would be unprotected."""
    covered = {entry["id"] for entry in _identities()}
    assert covered == set(BUILT_IN_MODEL_IDS)


def test_every_golden_identity_is_pinned_at_more_than_one_dimension() -> None:
    """The bucket a token lands in is ``hash(token) % dimensions``, so a single
    width would not catch a change that only moves wide vectors."""
    for model_id in BUILT_IN_MODEL_IDS:
        widths = {entry["dimensions"] for entry in _identities() if entry["id"] == model_id}
        assert len(widths) > 1, f"{model_id} is pinned at only {widths}"


@pytest.mark.parametrize("entry", _identities(), ids=_case_id)
def test_output_has_not_changed_under_this_revision(entry: dict[str, Any]) -> None:
    """If this fails, the embedder changed. Bump the revision; do not edit the fixture."""
    identity = ModelIdentity(id=entry["id"], revision=entry["revision"], dimensions=entry["dimensions"])
    embedder = build_built_in_embedder(identity)
    assert embedder.identity.triple == identity.triple
    for case in entry["vectors"]:
        produced = embedder.embed(case["text"])
        assert produced == case["values"], f"{_case_id(entry)} changed for {case['text']!r}"


@pytest.mark.parametrize("entry", _identities(), ids=_case_id)
def test_each_golden_vector_has_the_declared_width(entry: dict[str, Any]) -> None:
    for case in entry["vectors"]:
        assert len(case["values"]) == entry["dimensions"]


@pytest.mark.parametrize("entry", _identities(), ids=_case_id)
def test_a_golden_revision_is_one_this_release_implements(entry: dict[str, Any]) -> None:
    """A fixture entry for a revision that no longer exists must not pass silently."""
    build_built_in_embedder(ModelIdentity(id=entry["id"], revision=entry["revision"], dimensions=entry["dimensions"]))
