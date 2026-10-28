import json
from typing import Any

from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService


class ToolCallingService:
    def __init__(
        self,
        search_service: HybridSearchService,
        llm_service: LLMService,
    ):
        self.search_service = search_service
        self.llm_service = llm_service

    def _plan_follow_up(
        self,
        question: str,
        results: list[dict],
    ) -> str | None:
        evidence = "\n\n".join(
            f"Document: {result.get('title')}\n"
            f"Content: {result.get('content')}"
            for result in results
        )

        prompt = f"""
Original question:
{question}

Retrieved evidence:
{evidence}

Determine whether the evidence explicitly supports every fact
needed to answer the original question.

If an intermediate entity has been identified but another fact
about that entity is missing, request a follow-up search.

Return JSON only:
{{"query": "specific follow-up search query"}}

If the evidence is sufficient, return:
{{"query": null}}

Rules:
- Use only the retrieved evidence.
- Do not invent intermediate entities.
- Do not repeat the original question.
- Request only one follow-up query.
"""

        response = self.llm_service.generate(
            prompt=prompt,
            system_instruction=(
                "You are an evidence retrieval planner. "
                "Return valid JSON only."
            ),
            temperature=0.0,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Retrieval planner returned invalid JSON"
            ) from error

        if not isinstance(data, dict):
            raise RuntimeError("Invalid retrieval plan")

        query = data.get("query")

        if query is None:
            return None

        if not isinstance(query, str):
            raise RuntimeError("Invalid follow-up query")

        return query.strip() or None

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be between 1 and 3")

        tool_calls = []
        retrieved_results = []
        seen_chunks = set()
        executed_queries = set()

        def search_documents(query: str) -> dict:
            """Search indexed documents for evidence relevant to a question.

            Args:
                query: A concise search query for the needed information.
            """
            normalized_query = query.casefold().strip()

            if not normalized_query:
                return {
                    "sources": [],
                    "error": "Search query cannot be empty",
                }

            if normalized_query in executed_queries:
                return {
                    "sources": [],
                    "message": "Query already executed",
                }

            if len(tool_calls) >= max_hops:
                return {
                    "sources": [],
                    "message": "Search limit reached",
                }

            executed_queries.add(normalized_query)

            results = self.search_service.search(
                query=query,
                mode="hybrid",
                top_k=top_k,
            )

            sources = []

            for result in results:
                chunk_id = result.get("chunk_id")

                if chunk_id and chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    retrieved_results.append(result)

                sources.append(
                    {
                        "document_title": result.get("title"),
                        "content": result.get("content"),
                    }
                )

            tool_calls.append(
                {
                    "tool": "search_documents",
                    "hop": len(tool_calls) + 1,
                    "query": query,
                    "results_count": len(results),
                }
            )

            return {"sources": sources}

        current_query = question

        for hop in range(max_hops):
            previous_count = len(tool_calls)

            self.llm_service.generate_with_tools(
                prompt=(
                    f"Search the indexed documents for evidence "
                    f"relevant to this query:\n{current_query}\n\n"
                    "Use the search_documents tool."
                ),
                tools=[search_documents],
                system_instruction=(
                    "You are an evidence retrieval assistant. "
                    "Call search_documents to retrieve evidence. "
                    "Do not answer from general knowledge."
                ),
                temperature=0.0,
            )

            if len(tool_calls) == previous_count:
                break

            if hop == max_hops - 1:
                break

            next_query = self._plan_follow_up(
                question=question,
                results=retrieved_results,
            )
            print("Follow-up planner result:", next_query)

            if not next_query:
                break

            if next_query.casefold().strip() in executed_queries:
                break

            current_query = next_query

        return {
            "results": retrieved_results,
            "tool_calls": tool_calls,
        }