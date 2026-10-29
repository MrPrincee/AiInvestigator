from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService
from app.services.tool_calling_service import ToolCallingService

from app.services.rag.evidence import (
    build_sources,
    build_context,
)
from app.services.rag.response_parser import (
    parse_json_response,
    parse_response,
    parse_planning_response,
)
from app.services.rag.response_validator import validate_claims
from app.services.rag.retrieval_planner import RetrievalPlanner


class RAGService:
    SYSTEM_INSTRUCTION = """
You are an evidence-based investigation assistant.
Answer using only the supplied evidence, never outside knowledge.
Return a JSON object with these fields:
{
  "is_answerable": true,
  "claims": [
    {"text": "A factual statement without citation markers.", "source_ids": [1]}
  ],
  "has_contradictions": false,
  "contradictions": []
}
Every factual claim must cite the IDs of evidence blocks that directly support it.
For a multi-step conclusion, cite every evidence block needed to establish the chain.
Never write citation markers inside claim text; the application adds them.
Use only IDs shown in the evidence. Do not invent facts or citations.
If evidence is insufficient, return is_answerable false and claims [].
If sources conflict, set has_contradictions true and describe each conflict in contradictions.
Each contradiction should be a JSON object with "description" (string) and "source_ids" (list of evidence IDs).
For conflicting answers, set is_answerable false, claims [], and provide contradictions with both source IDs.
Do not silently choose one conflicting value.
Return valid JSON only, with no markdown or commentary.
"""

    def __init__(
        self,
        search_service: HybridSearchService,
        llm_service: LLMService,
    ):
        self.search_service = search_service
        self.llm_service = llm_service

        self.tool_calling_service = ToolCallingService(
            search_service,
            llm_service,
        )

        self.retrieval_planner = RetrievalPlanner(
            search_service=search_service,
            llm_service=llm_service,
        )

    # Backward-compatible helpers

    @staticmethod
    def _build_sources(results: list[dict]) -> list[dict]:
        return build_sources(results)

    @staticmethod
    def _build_context(sources: list[dict]) -> str:
        return build_context(sources)

    _parse_json_response = staticmethod(parse_json_response)
    _parse_response = staticmethod(parse_response)
    _parse_planning_response = staticmethod(parse_planning_response)
    _validate_claims = staticmethod(validate_claims)

    def _plan_follow_up_queries(
        self,
        question: str,
        sources: list[dict],
    ) -> list[str]:
        return self.retrieval_planner.plan_follow_up_queries(
            question=question,
            sources=sources,
        )

    def _retrieve_multi_hop(
        self,
        question: str,
        top_k: int,
        max_hops: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        return self.retrieval_planner.retrieve_multi_hop(
            question=question,
            top_k=top_k,
            max_hops=max_hops,
        )

    @staticmethod
    def _abstention_response(
        question: str,
        retrieved_sources: list[dict] | None = None,
        search_history: list[dict] | None = None,
    ) -> dict:
        return {
            "question": question,
            "answer": (
                "The available documents do not provide "
                "sufficient evidence to answer this question."
            ),
            "is_answerable": False,
            "has_contradictions": False,
            "contradictions": [],
            "sources": [],
            "retrieved_sources": retrieved_sources or [],
            "search_history": search_history or [],
        }

    def answer(
        self,
        question: str,
        top_k: int = 8,
        max_hops: int = 3,
        retrieval_mode: str = "standard",
    ) -> dict:
        if retrieval_mode not in ("standard", "tool_calling"):
            raise ValueError(
                "retrieval_mode must be 'standard' or 'tool_calling'"
            )

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be between 1 and 3")

        # Step 1: Retrieve evidence

        if retrieval_mode == "standard":
            sources, search_history = self._retrieve_multi_hop(
                question=question,
                top_k=top_k,
                max_hops=max_hops,
            )

        else:
            retrieval = self.tool_calling_service.retrieve(
                question=question,
                top_k=top_k,
                max_hops=max_hops,
            )

            sources = self._build_sources(
                retrieval["results"]
            )

            search_history = [
                {
                    "hop": call["hop"],
                    "query": call["query"],
                    "results_count": call["results_count"],
                }
                for call in retrieval["tool_calls"]
            ]

        # Step 2: Handle missing evidence

        if not sources:
            return self._abstention_response(
                question=question,
                search_history=search_history,
            )

        # Step 3: Prepare evidence

        available_sources = {
            source["citation_id"]: source
            for source in sources
        }

        prompt = (
            f"Question:\n{question}\n\n"
            f"Available evidence:\n{self._build_context(sources)}\n\n"
            "Return the required JSON. "
            "Each claim must be grounded in its cited evidence."
        )

        # Step 4: Generate and validate answer

        last_error = None

        for attempt in range(2):
            raw_response = self.llm_service.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.0,
            )

            try:
                data = self._parse_response(raw_response)

                self._validate_claims(
                    data,
                    available_sources,
                )

                break

            except RuntimeError as error:
                last_error = error

                prompt = (
                    f"{prompt}\n\n"
                    f"Your previous output was invalid: {error}. "
                    "Regenerate the complete JSON with valid claims "
                    "and existing source IDs."
                )

        else:
            raise RuntimeError(
                "LLM failed to return a valid evidence-grounded "
                "response after retry"
            ) from last_error

        # Step 5: Process contradictions

        contradiction_ids = list(
            dict.fromkeys(
                source_id
                for contradiction in data["contradictions"]
                for source_id in contradiction["source_ids"]
            )
        )

        contradiction_details = [
            contradiction["description"]
            for contradiction in data["contradictions"]
        ]

        # Step 6: Handle unanswerable questions

        if not data["is_answerable"]:
            response = self._abstention_response(
                question=question,
                retrieved_sources=sources,
                search_history=search_history,
            )

            if data["has_contradictions"]:
                response["answer"] = (
                    "The available documents contain "
                    "conflicting evidence: "
                    + " ".join(contradiction_details)
                )

            response["has_contradictions"] = (
                data["has_contradictions"]
            )

            response["contradictions"] = contradiction_details

            response["sources"] = [
                available_sources[source_id]
                for source_id in contradiction_ids
            ]

            return response

        # Step 7: Format cited answer

        answer_parts = []
        cited_ids = []

        for claim in data["claims"]:
            unique_ids = list(
                dict.fromkeys(claim["source_ids"])
            )

            for source_id in unique_ids:
                if source_id not in cited_ids:
                    cited_ids.append(source_id)

            citation_text = "".join(
                f"[{source_id}]"
                for source_id in unique_ids
            )

            claim_text = claim["text"].strip()

            answer_parts.append(
                f"{claim_text} {citation_text}"
            )

        for source_id in contradiction_ids:
            if source_id not in cited_ids:
                cited_ids.append(source_id)

        # Step 8: Return final response

        return {
            "question": question,
            "answer": " ".join(answer_parts),
            "is_answerable": True,
            "has_contradictions": data["has_contradictions"],
            "contradictions": contradiction_details,
            "sources": [
                available_sources[source_id]
                for source_id in cited_ids
            ],
            "retrieved_sources": sources,
            "search_history": search_history,
        }