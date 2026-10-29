from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService

from app.services.rag.evidence import (
    build_context,
    build_sources,
)
from app.services.rag.response_parser import (
    parse_planning_response,
)


class RetrievalPlanner:
    def __init__(
        self,
        search_service: HybridSearchService,
        llm_service: LLMService,
    ):
        self.search_service = search_service
        self.llm_service = llm_service

    def plan_follow_up_queries(
        self,
        question: str,
        sources: list[dict],
    ) -> list[str]:
        context = build_context(sources)

        prompt = f"""
Original question:
{question}

Evidence already retrieved:
{context}

Determine whether additional document searches are necessary
to answer the original question.

Focus on missing facts, intermediate entities, relationships,
dates, and evidence from other documents.

Return only valid JSON in this format:

{{
    "queries": [
        "first follow-up search query",
        "second follow-up search query"
    ]
}}

Rules:
- Return at most 2 queries.
- Return an empty list if the evidence is sufficient.
- Do not repeat the original question.
- Do not invent facts.
- Search queries should be concise and specific.
"""

        response = self.llm_service.generate(
            prompt=prompt,
            system_instruction=(
                "You are a retrieval planning assistant. "
                "Return only valid JSON."
            ),
            temperature=0.0,
        )

        data = parse_planning_response(response)

        return data["queries"]

    def retrieve_multi_hop(
        self,
        question: str,
        top_k: int,
        max_hops: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        if not 1 <= max_hops <= 3:
            raise ValueError(
                "max_hops must be between 1 and 3"
            )

        all_results = []
        search_history = []
        executed_queries = set()

        initial_results = self.search_service.search(
            query=question,
            mode="hybrid",
            top_k=top_k,
        )

        all_results.extend(initial_results)

        executed_queries.add(
            question.casefold().strip()
        )

        search_history.append(
            {
                "hop": 1,
                "query": question,
                "results_count": len(initial_results),
            }
        )

        for hop in range(2, max_hops + 1):
            current_sources = build_sources(all_results)

            if not current_sources:
                break

            follow_up_queries = self.plan_follow_up_queries(
                question=question,
                sources=current_sources,
            )

            if not follow_up_queries:
                break

            new_searches = 0

            for query in follow_up_queries:
                normalized_query = query.casefold().strip()

                if normalized_query in executed_queries:
                    continue

                executed_queries.add(normalized_query)

                results = self.search_service.search(
                    query=query,
                    mode="hybrid",
                    top_k=top_k,
                )

                all_results.extend(results)
                new_searches += 1

                search_history.append(
                    {
                        "hop": hop,
                        "query": query,
                        "results_count": len(results),
                    }
                )

            if new_searches == 0:
                break

        return (
            build_sources(all_results),
            search_history,
        )