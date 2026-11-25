"""Vectors of different dimensions are never compared silently.

Before this fix ``cosine_similarity`` returned 0.0 for a length mismatch, so a
store holding vectors of another dimension ranked them last with no error.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_ai_stack.api.app import create_app
from private_ai_stack.config.settings import get_settings
from private_ai_stack.memory.embeddings import EmbeddingDimensionError, cosine_similarity
from private_ai_stack.memory.models import DocumentChunk

PLANTED = "cos-planted-0b5d-do-not-log"


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(EmbeddingDimensionError) as raised:
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    assert "2-dimension" in str(raised.value) and "3-dimension" in str(raised.value)
    assert isinstance(raised.value, ValueError)


def test_matching_lengths_still_score() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_an_empty_pair_still_scores_zero() -> None:
    assert cosine_similarity([], []) == 0.0


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", "memory://local")
    monkeypatch.setenv("AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("EXPORT_DIR", str(tmp_path / "exports"))
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()


def _plant_foreign_chunk(client: TestClient) -> None:
    """Add a chunk embedded at another dimension, as a mixed or misconfigured store would hold."""
    store = client.app.state.memory  # type: ignore[attr-defined]
    store._chunks.append(
        DocumentChunk(
            document_id="doc-foreign",
            chunk_id="doc-foreign-0000",
            source_name="foreign.md",
            text=PLANTED,
            content_hash="0" * 64,
            embedding=[0.5] * 32,
        )
    )


def test_search_over_a_mixed_store_is_an_explicit_conflict(client: TestClient) -> None:
    ingest = client.post("/v1/knowledge/documents", json={"source_name": "doc.md", "content": "audit records are portable"})
    assert ingest.status_code == 200
    _plant_foreign_chunk(client)

    response = client.post("/v1/knowledge/search", json={"query": "audit", "limit": 5})

    assert response.status_code == 409
    assert response.json()["error"] == "knowledge_index_inconsistent"
    assert PLANTED not in response.text


def test_a_consistent_store_still_searches(client: TestClient) -> None:
    client.post("/v1/knowledge/documents", json={"source_name": "doc.md", "content": "audit records are portable"})
    response = client.post("/v1/knowledge/search", json={"query": "audit", "limit": 1})
    assert response.status_code == 200
    assert response.json()["hits"][0]["source_name"] == "doc.md"
