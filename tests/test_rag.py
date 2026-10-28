import pytest

from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


QUESTION = (
    "In what year was the actress who portrayed "
    "Beth Greene in The Walking Dead born?"
)


@pytest.fixture(scope="module")
def rag_service():
    search_service = HybridSearchService()
    llm_service = LLMService()

    try:
        yield RAGService(
            search_service=search_service,
            llm_service=llm_service,
        )
    finally:
        llm_service.close()
        search_service.close()


@pytest.mark.parametrize(
    "retrieval_mode",
    ["standard", "tool_calling"],
)
def test_rag_answer(rag_service, retrieval_mode):
    result = rag_service.answer(
        question=QUESTION,
        top_k=5,
        max_hops=3,
        retrieval_mode=retrieval_mode,
    )

    assert result["question"] == QUESTION
    assert result["is_answerable"] is True
    assert result["has_contradictions"] is False
    assert "1985" in result["answer"]

    assert len(result["sources"]) > 0
    assert len(result["retrieved_sources"]) > 0
    assert len(result["search_history"]) > 0

    cited_titles = {
        source["document_title"]
        for source in result["sources"]
    }

    assert "Beth Greene" in cited_titles
    assert "Emily Kinney" in cited_titles

    for source in result["sources"]:
        assert source["citation_id"] > 0
        assert source["document_id"]
        assert source["chunk_id"]
        assert source["evidence"]


@pytest.mark.parametrize(
    "retrieval_mode",
    ["standard", "tool_calling"],
)
def test_rag_unanswerable_question(rag_service, retrieval_mode):
    result = rag_service.answer(
        question=(
            "What is the private password of the actress "
            "who portrayed Beth Greene in The Walking Dead?"
        ),
        top_k=5,
        max_hops=3,
        retrieval_mode=retrieval_mode,
    )

    assert result["is_answerable"] is False
    assert result["sources"] == []
    assert result["has_contradictions"] is False


@pytest.mark.parametrize(
    "retrieval_mode",
    ["standard", "tool_calling"],
)
def test_rag_invalid_question(rag_service, retrieval_mode):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag_service.answer(
            question="",
            retrieval_mode=retrieval_mode,
        )


def test_rag_invalid_retrieval_mode(rag_service):
    with pytest.raises(ValueError):
        rag_service.answer(
            question=QUESTION,
            retrieval_mode="invalid_mode",
        )