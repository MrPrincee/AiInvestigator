import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeHybridSearchService:
    def search(self, query, mode="hybrid", top_k=5):
        if mode not in {"semantic", "keyword", "hybrid"}:
            raise ValueError(f"Unsupported search mode: {mode}")

        return [
            {
                "chunk_id": "test-chunk-1",
                "document_id": "test-document-1",
                "content": "Python is a programming language.",
                "title": "Python",
                "source": "test",
                "score": 0.03,
            }
        ][:top_k]


@pytest.fixture
def client():
    previous_service = getattr(app.state, "search_service", None)

    app.state.search_service = FakeHybridSearchService()

    try:
        yield TestClient(app)
    finally:
        if previous_service is None:
            del app.state.search_service
        else:
            app.state.search_service = previous_service


def test_search_endpoint(client):
    response = client.post(
        "/search",
        json={
            "query": "Python",
            "mode": "hybrid",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "Python"
    assert data["mode"] == "hybrid"
    assert data["total_results"] == 1
    assert data["results"][0]["chunk_id"] == "test-chunk-1"


def test_search_invalid_mode(client):
    response = client.post(
        "/search",
        json={
            "query": "Python",
            "mode": "invalid",
            "top_k": 3,
        },
    )

    assert response.status_code == 400


def test_search_invalid_top_k(client):
    response = client.post(
        "/search",
        json={
            "query": "Python",
            "mode": "hybrid",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_search_missing_query(client):
    response = client.post(
        "/search",
        json={
            "mode": "hybrid",
            "top_k": 3,
        },
    )

    assert response.status_code == 422