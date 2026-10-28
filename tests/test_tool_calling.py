import pytest

from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService
from app.services.tool_calling_service import ToolCallingService


@pytest.fixture(scope="module")
def tool_service():
    search_service = HybridSearchService()
    llm_service = LLMService()

    try:
        yield ToolCallingService(
            search_service=search_service,
            llm_service=llm_service,
        )
    finally:
        llm_service.close()
        search_service.close()


def test_tool_calling_multi_hop(tool_service):
    question = (
        "What is the birthplace of the actress "
        "who portrayed Beth Greene in The Walking Dead?"
    )

    result = tool_service.retrieve(
        question=question,
        top_k=3,
        max_hops=3,
    )

    assert "results" in result
    assert "tool_calls" in result

    assert len(result["tool_calls"]) >= 1
    assert len(result["tool_calls"]) <= 3
    assert len(result["results"]) > 0

    for call in result["tool_calls"]:
        assert call["tool"] == "search_documents"
        assert call["query"]
        assert call["results_count"] >= 0

    document_titles = {
        item.get("title")
        for item in result["results"]
    }

    assert "Beth Greene" in document_titles


@pytest.mark.parametrize(
    "question",
    ["", "   "],
)
def test_tool_calling_empty_question(tool_service, question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        tool_service.retrieve(question=question)


def test_tool_calling_invalid_top_k(tool_service):
    with pytest.raises(ValueError, match="top_k"):
        tool_service.retrieve(
            question="Who portrayed Beth Greene?",
            top_k=0,
        )


def test_tool_calling_invalid_max_hops(tool_service):
    with pytest.raises(ValueError, match="max_hops"):
        tool_service.retrieve(
            question="Who portrayed Beth Greene?",
            max_hops=4,
        )