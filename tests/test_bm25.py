import pytest

from app.services.bm25_service import BM25Service


def test_bm25_finds_relevant_document():
    service = BM25Service()

    service.fit([
        {"id": "1", "content": "Python is a programming language."},
        {"id": "2", "content": "Paris is the capital of France."},
        {"id": "3", "content": "Python is widely used in machine learning."},
    ])

    results = service.search("Python programming", top_k=2)

    assert results[0]["id"] == "1"
    assert results[0]["score"] > 0


def test_bm25_returns_empty_for_unknown_terms():
    service = BM25Service()

    service.fit([
        {"id": "1", "content": "Python programming language."},
    ])

    assert service.search("astronomy") == []


def test_bm25_empty_collection():
    service = BM25Service()

    assert service.search("Python") == []


def test_bm25_invalid_parameters():
    with pytest.raises(ValueError):
        BM25Service(k1=0)

    with pytest.raises(ValueError):
        BM25Service(b=1.5)


def test_bm25_invalid_top_k():
    service = BM25Service()

    with pytest.raises(ValueError):
        service.search("Python", top_k=0)