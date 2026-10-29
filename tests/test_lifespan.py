from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture
def fake_services(monkeypatch):
    created = {"qdrant": [], "embedding": [], "search": [], "crud": [], "llm": [], "rag": []}

    class FakeQdrantService:
        def __init__(self):
            self.close = Mock()
            created["qdrant"].append(self)

    class FakeEmbeddingService:
        def __init__(self):
            created["embedding"].append(self)

    class FakeHybridSearchService:
        def __init__(self, *, qdrant_service, embedding_service):
            self.qdrant_service = qdrant_service
            self.embedding_service = embedding_service
            created["search"].append(self)

    class FakeDocumentCRUDService:
        def __init__(self, *, qdrant_service, embedding_service):
            self.qdrant_service = qdrant_service
            self.embedding_service = embedding_service
            created["crud"].append(self)

    class FakeLLMService:
        def __init__(self):
            self.close = Mock()
            created["llm"].append(self)

    class FakeRAGService:
        def __init__(self, *, search_service, llm_service):
            self.search_service = search_service
            self.llm_service = llm_service
            created["rag"].append(self)

    monkeypatch.setattr(main_module, "QdrantService", FakeQdrantService)
    monkeypatch.setattr(main_module, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(main_module, "HybridSearchService", FakeHybridSearchService)
    monkeypatch.setattr(main_module, "DocumentCRUDService", FakeDocumentCRUDService)
    monkeypatch.setattr(main_module, "LLMService", FakeLLMService)
    monkeypatch.setattr(main_module, "RAGService", FakeRAGService)
    return created


def test_lifespan_shares_resources_and_closes_once(fake_services):
    with TestClient(main_module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        state = main_module.app.state
        assert len(fake_services["qdrant"]) == 1
        assert len(fake_services["embedding"]) == 1
        assert state.search_service.qdrant_service is fake_services["qdrant"][0]
        assert state.crud_service.qdrant_service is fake_services["qdrant"][0]
        assert state.search_service.embedding_service is fake_services["embedding"][0]
        assert state.crud_service.embedding_service is fake_services["embedding"][0]
        assert state.rag_service.search_service is state.search_service
        assert state.rag_service.llm_service is state.llm_service
        fake_services["qdrant"][0].close.assert_not_called()
        fake_services["llm"][0].close.assert_not_called()

    fake_services["qdrant"][0].close.assert_called_once_with()
    fake_services["llm"][0].close.assert_called_once_with()


def test_lifespan_closes_qdrant_if_embedding_initialization_fails(fake_services, monkeypatch):
    def fail_embedding():
        raise RuntimeError("Embedding initialization failed")

    monkeypatch.setattr(main_module, "EmbeddingService", fail_embedding)

    with pytest.raises(RuntimeError, match="Embedding initialization failed"):
        with TestClient(main_module.app):
            pass

    assert len(fake_services["qdrant"]) == 1
    fake_services["qdrant"][0].close.assert_called_once_with()
    assert fake_services["llm"] == []


def test_lifespan_closes_qdrant_if_llm_initialization_fails(fake_services, monkeypatch):
    def fail_llm():
        raise RuntimeError("LLM initialization failed")

    monkeypatch.setattr(main_module, "LLMService", fail_llm)

    with pytest.raises(RuntimeError, match="LLM initialization failed"):
        with TestClient(main_module.app):
            pass

    assert len(fake_services["qdrant"]) == 1
    fake_services["qdrant"][0].close.assert_called_once_with()
    assert fake_services["llm"] == []


def test_lifespan_closes_both_if_rag_initialization_fails(fake_services, monkeypatch):
    def fail_rag(*, search_service, llm_service):
        raise RuntimeError("RAG initialization failed")

    monkeypatch.setattr(main_module, "RAGService", fail_rag)

    with pytest.raises(RuntimeError, match="RAG initialization failed"):
        with TestClient(main_module.app):
            pass

    fake_services["qdrant"][0].close.assert_called_once_with()
    fake_services["llm"][0].close.assert_called_once_with()
