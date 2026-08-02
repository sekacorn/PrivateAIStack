from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from private_ai_stack.api.app import create_app
from private_ai_stack.config.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_health_version_and_policies(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200
    policies = client.get("/v1/policies").json()["policies"]
    assert policies


def test_knowledge_ingest_and_search(client: TestClient) -> None:
    response = client.post("/v1/knowledge/documents", json={"source_name": "doc.md", "content": "audit records are portable"})
    assert response.status_code == 200
    search = client.post("/v1/knowledge/search", json={"query": "audit", "limit": 1})
    assert search.status_code == 200
    assert search.json()["hits"][0]["source_name"] == "doc.md"


def test_review_sample_target(client: TestClient) -> None:
    response = client.post(
        "/v1/reviews",
        json={"repository_path": str(PROJECT_ROOT / "sample-target"), "mode": "safe-static"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["summary"]["source_unchanged"] is True
