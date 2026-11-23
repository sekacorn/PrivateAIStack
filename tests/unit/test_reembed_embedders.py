"""The built-in embedders, and the claim that they match what they reimplement."""

from __future__ import annotations

import asyncio

import pytest
from forge.memory.vector import InMemoryVectorStore

from private_ai_stack.memory.embeddings import DeterministicEmbeddingModel
from private_ai_stack.reembed.embedders import (
    AGENTFORGE_HASHED_BOW,
    BUILT_IN_MODEL_IDS,
    PRIVATEAISTACK_HASHED_BOW,
    AgentForgeHashedBagOfWords,
    EmbedderError,
    ModelIdentity,
    PrivateAIStackSignedHashedBagOfWords,
    build_built_in_embedder,
)

TEXTS = [
    "Reset the password through the self-service portal.",
    "Refunds over 500 need a supervisor approval and an audit entry.",
    "Escalate to the on-call engineer when the queue depth exceeds 40.",
    "",
    "   ",
    "Ünïcödé tokens 123 stay deterministic",
]


def identity(model_id: str, dimensions: int = 64, revision: str = "1") -> ModelIdentity:
    return ModelIdentity(id=model_id, revision=revision, dimensions=dimensions)


@pytest.mark.parametrize(
    ("field", "value"),
    [("id", ""), ("id", "x" * 129), ("revision", ""), ("revision", "x" * 129)],
)
def test_identity_refuses_unusable_strings(field: str, value: str) -> None:
    fields = {"id": "m", "revision": "1", "dimensions": 8, field: value}
    with pytest.raises(EmbedderError):
        ModelIdentity(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize("dimensions", [0, -1, 8193])
def test_identity_refuses_dimensions_outside_the_format(dimensions: int) -> None:
    with pytest.raises(EmbedderError, match="between 1 and 8192"):
        identity(PRIVATEAISTACK_HASHED_BOW, dimensions)


def test_identity_is_a_triple() -> None:
    assert identity(PRIVATEAISTACK_HASHED_BOW, 64).triple == (PRIVATEAISTACK_HASHED_BOW, "1", 64)


def test_both_built_in_schemes_are_available() -> None:
    assert BUILT_IN_MODEL_IDS == (AGENTFORGE_HASHED_BOW, PRIVATEAISTACK_HASHED_BOW)


@pytest.mark.parametrize("model_id", BUILT_IN_MODEL_IDS)
def test_built_in_embedder_reports_the_identity_it_was_asked_for(model_id: str) -> None:
    assert build_built_in_embedder(identity(model_id, 128)).identity.triple == (model_id, "1", 128)


def test_unknown_model_id_is_refused_rather_than_approximated() -> None:
    with pytest.raises(EmbedderError, match="not a built-in embedder"):
        build_built_in_embedder(identity("some-vendor-embedding-v3"))


def test_known_id_at_an_unimplemented_revision_is_refused() -> None:
    with pytest.raises(EmbedderError, match="revision '1', not '2'"):
        build_built_in_embedder(identity(PRIVATEAISTACK_HASHED_BOW, revision="2"))


@pytest.mark.parametrize("model_id", BUILT_IN_MODEL_IDS)
@pytest.mark.parametrize("text", TEXTS)
def test_embedding_is_deterministic_and_has_the_declared_width(model_id: str, text: str) -> None:
    embedder = build_built_in_embedder(identity(model_id, 32))
    first = embedder.embed(text)
    assert first == embedder.embed(text)
    assert len(first) == 32


@pytest.mark.parametrize("model_id", BUILT_IN_MODEL_IDS)
def test_dimension_is_part_of_identity_because_it_changes_every_component(model_id: str) -> None:
    narrow = build_built_in_embedder(identity(model_id, 16)).embed(TEXTS[0])
    wide = build_built_in_embedder(identity(model_id, 64)).embed(TEXTS[0])
    assert len(narrow) != len(wide)
    assert narrow[:16] != wide[:16]


@pytest.mark.parametrize("model_id", BUILT_IN_MODEL_IDS)
def test_text_with_no_tokens_embeds_to_zero_rather_than_failing(model_id: str) -> None:
    assert build_built_in_embedder(identity(model_id, 8)).embed("   ") == [0.0] * 8


@pytest.mark.parametrize("text", TEXTS)
def test_privateaistack_scheme_matches_the_shipped_memory_model(text: str) -> None:
    """Component for component against the model the memory store actually uses."""
    assert PrivateAIStackSignedHashedBagOfWords(64).embed(text) == DeterministicEmbeddingModel(64).embed(text)


def test_agentforge_scheme_matches_agentforge_through_its_public_search_scores() -> None:
    """A black-box check: AgentForge's own cosine scores must equal ours.

    ``InMemoryVectorStore`` keeps its vectors and its dimension private, so the
    comparison goes through the public API it does expose. AgentForge rounds
    the score it reports to six decimal places, so the assertion does too.
    """
    store = InMemoryVectorStore(dim=512)
    for text in TEXTS[:3]:
        asyncio.run(store.add(text))
    ours = AgentForgeHashedBagOfWords(512)
    query = "supervisor approval for refunds"
    query_vector = ours.embed(query)
    expected = {text: round(sum(a * b for a, b in zip(query_vector, ours.embed(text), strict=True)), 6) for text in TEXTS[:3]}
    results = asyncio.run(store.search(query, k=3))
    assert results, "AgentForge returned no hit, so the comparison would prove nothing"
    for item in results:
        assert item.score == pytest.approx(expected[item.text])


def test_the_two_schemes_disagree_so_re_embedding_between_them_is_a_real_migration() -> None:
    agentforge = AgentForgeHashedBagOfWords(64).embed(TEXTS[0])
    privateaistack = PrivateAIStackSignedHashedBagOfWords(64).embed(TEXTS[0])
    assert agentforge != privateaistack
