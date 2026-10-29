import pytest

from app.services.rag_service import RAGService
import json

def test_rejects_unknown_citation_id():
    data = {
        "is_answerable": True,
        "claims": [
            {
                "text": "Emily Kinney was born in 1985.",
                "source_ids": [99],
            }
        ],
        "has_contradictions": False,
        "contradictions": [],
    }

    available_sources = {
        1: {
            "citation_id": 1,
            "evidence": "Emily Kinney was born in 1985.",
        }
    }

    with pytest.raises(RuntimeError, match="unknown source"):
        RAGService._validate_claims(
            data=data,
            available_sources=available_sources,
        )


def test_rejects_answer_with_contradictory_evidence():
    data = {
        "is_answerable": True,
        "claims": [
            {
                "text": "The system was updated on March 10.",
                "source_ids": [1],
            }
        ],
        "has_contradictions": True,
        "contradictions": [
            {
                "description": (
                    "Source 1 reports March 10, "
                    "but source 2 reports March 12."
                ),
                "source_ids": [1, 2],
            }
        ],
    }

    available_sources = {
        1: {
            "citation_id": 1,
            "evidence": "The system was updated on March 10.",
        },
        2: {
            "citation_id": 2,
            "evidence": "The system was updated on March 12.",
        },
    }

    with pytest.raises(RuntimeError, match="Conflicting evidence"):
        RAGService._validate_claims(
            data=data,
            available_sources=available_sources,
        )


def test_rejects_invalid_json():
    invalid_response = """
    {
        "is_answerable": true,
        "claims": [
    """

    with pytest.raises(RuntimeError, match="LLM returned invalid JSON"):
        RAGService._parse_response(invalid_response)


def test_abstention_when_evidence_is_insufficient():
    question = "What is the administrator's private password?"

    result = RAGService._abstention_response(
        question=question,
        retrieved_sources=[],
        search_history=[],
    )

    assert result["question"] == question
    assert result["is_answerable"] is False
    assert result["has_contradictions"] is False
    assert result["sources"] == []
    assert result["retrieved_sources"] == []
    assert result["search_history"] == []
    assert "sufficient evidence" in result["answer"]





class FakeSearchService:
    def search(self, query, mode, top_k):
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "title": "Emily Kinney",
                "source": "test",
                "content": "Emily Kinney was born in 1985.",
                "chunk_index": 0,
                "page_number": 1,
                "score": 0.95,
            }
        ]


class FakeLLMService:
    def generate(self, prompt, system_instruction, temperature):
        return json.dumps(
            {
                "is_answerable": True,
                "claims": [
                    {
                        "text": "Emily Kinney was born in 1985.",
                        "source_ids": [1],
                    }
                ],
                "has_contradictions": False,
                "contradictions": [],
            }
        )


def test_rag_answer_with_fake_services():
    search_service = FakeSearchService()
    llm_service = FakeLLMService()

    rag_service = RAGService(
        search_service=search_service,
        llm_service=llm_service,
    )

    result = rag_service.answer(
        question="When was Emily Kinney born?",
        top_k=5,
        max_hops=1,
        retrieval_mode="standard",
    )

    assert result["is_answerable"] is True
    assert result["has_contradictions"] is False
    assert "1985" in result["answer"]
    assert "[1]" in result["answer"]

    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "doc-1"
    assert result["sources"][0]["citation_id"] == 1

    assert len(result["search_history"]) == 1
    assert result["search_history"][0]["hop"] == 1


class FakeMultiHopSearchService:
    def __init__(self):
        self.queries = []

    def search(self, query, mode, top_k):
        self.queries.append(query)

        if query == "Who portrayed Beth Greene?":
            return [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "title": "Beth Greene",
                    "source": "test",
                    "content": "Beth Greene was portrayed by Emily Kinney.",
                    "chunk_index": 0,
                    "page_number": 1,
                    "score": 0.95,
                }
            ]

        if query == "Emily Kinney birth year":
            return [
                {
                    "chunk_id": "chunk-2",
                    "document_id": "doc-2",
                    "title": "Emily Kinney",
                    "source": "test",
                    "content": "Emily Kinney was born in 1985.",
                    "chunk_index": 0,
                    "page_number": 1,
                    "score": 0.90,
                }
            ]

        return []


class FakeMultiHopLLMService:
    def generate(self, prompt, system_instruction, temperature):
        if "retrieval planning assistant" in system_instruction:
            return json.dumps({
                "queries": ["Emily Kinney birth year"]
            })

        return json.dumps({
            "is_answerable": True,
            "claims": [
                {
                    "text": "Emily Kinney, who portrayed Beth Greene, was born in 1985.",
                    "source_ids": [1, 2],
                }
            ],
            "has_contradictions": False,
            "contradictions": [],
        })


def test_rag_multi_hop_with_fake_services():
    search_service = FakeMultiHopSearchService()
    llm_service = FakeMultiHopLLMService()

    rag_service = RAGService(
        search_service=search_service,
        llm_service=llm_service,
    )

    result = rag_service.answer(
        question="Who portrayed Beth Greene?",
        top_k=5,
        max_hops=2,
        retrieval_mode="standard",
    )

    assert result["is_answerable"] is True
    assert "1985" in result["answer"]

    assert len(result["sources"]) == 2
    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]

    assert len(result["search_history"]) == 2

    assert search_service.queries == [
        "Who portrayed Beth Greene?",
        "Emily Kinney birth year",
    ]


def test_rag_tool_calling_with_fake_services():
    search_service = FakeSearchService()
    llm_service = FakeLLMService()

    rag_service = RAGService(
        search_service=search_service,
        llm_service=llm_service,
    )

    class FakeToolCallingService:
        def retrieve(self, question, top_k, max_hops):
            return {
                "results": search_service.search(
                    query=question,
                    mode="hybrid",
                    top_k=top_k,
                ),
                "tool_calls": [
                    {
                        "tool": "search_documents",
                        "hop": 1,
                        "query": question,
                        "results_count": 1,
                    }
                ],
            }

    rag_service.tool_calling_service = FakeToolCallingService()

    result = rag_service.answer(
        question="When was Emily Kinney born?",
        top_k=5,
        max_hops=2,
        retrieval_mode="tool_calling",
    )

    assert result["is_answerable"] is True
    assert result["has_contradictions"] is False
    assert "1985" in result["answer"]
    assert "[1]" in result["answer"]

    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "doc-1"

    assert len(result["search_history"]) == 1
    assert result["search_history"][0]["hop"] == 1