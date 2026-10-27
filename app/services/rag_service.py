import json
import re
from typing import Any

from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService


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

    @staticmethod
    def _build_sources(results: list[dict]) -> list[dict]:
        sources = []
        seen_chunks = set()

        for result in results:
            chunk_id = result.get("chunk_id")
            document_id = result.get("document_id")
            content = result.get("content")

            if not chunk_id or not document_id or not content:
                continue

            if chunk_id in seen_chunks:
                continue

            seen_chunks.add(chunk_id)

            sources.append(
                {
                    "citation_id": len(sources) + 1,
                    "document_id": document_id,
                    "document_title": result.get("title"),
                    "source": result.get("source"),
                    "chunk_id": chunk_id,
                    "chunk_index": result.get("chunk_index"),
                    "page_number": result.get("page_number"),
                    "evidence": content,
                    "retrieval_score": result.get("score"),
                }
            )

        return sources

    @staticmethod
    def _build_context(sources: list[dict]) -> str:
        blocks = []

        for source in sources:
            blocks.append(
                f"[{source['citation_id']}]\n"
                f"Document: {source['document_title']}\n"
                f"Document ID: {source['document_id']}\n"
                f"Page: {source['page_number']}\n"
                f"Evidence:\n{source['evidence']}"
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _parse_json_response(response: str) -> dict[str, Any]:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "LLM returned invalid JSON"
            ) from error

        if not isinstance(data, dict):
            raise RuntimeError(
                "LLM response must be a JSON object"
            )

        return data

    @classmethod
    def _parse_response(cls, response: str) -> dict[str, Any]:
        data = cls._parse_json_response(response)
        required = {
            "is_answerable",
            "claims",
            "has_contradictions",
            "contradictions",
        }
        if not required.issubset(data):
            raise RuntimeError("LLM response is missing required fields")
        if type(data["is_answerable"]) is not bool:
            raise RuntimeError("Invalid is_answerable field")
        if type(data["has_contradictions"]) is not bool:
            raise RuntimeError("Invalid has_contradictions field")
        if not isinstance(data["claims"], list):
            raise RuntimeError("Invalid claims field")
        contradictions = data["contradictions"]
        if isinstance(contradictions, dict):
            contradictions = [contradictions]
        if not isinstance(contradictions, list):
            raise RuntimeError("Invalid contradictions field")
        normalized = []
        for item in contradictions:
            if isinstance(item, str) and item.strip():
                normalized.append({"description": item.strip(), "source_ids": []})
            elif isinstance(item, dict):
                description = next(
                    (
                        item[key].strip()
                        for key in ("description", "explanation", "conflict", "text", "details")
                        if isinstance(item.get(key), str) and item[key].strip()
                    ),
                    None,
                )
                if description is None:
                    parts = [
                        f"{key}: {value}"
                        for key, value in item.items()
                        if key not in ("source_ids", "sources", "citation_ids")
                    ]
                    description = "; ".join(parts).strip()
                ids = item.get("source_ids", item.get("citation_ids", []))
                if not isinstance(ids, list) or not all(type(value) is int for value in ids):
                    raise RuntimeError("Invalid contradiction source_ids")
                if not description:
                    raise RuntimeError("Invalid contradiction description")
                normalized.append({"description": description, "source_ids": list(dict.fromkeys(ids))})
            else:
                raise RuntimeError("Invalid contradiction entry")
        data["contradictions"] = normalized
        if data["has_contradictions"] and not normalized:
            raise RuntimeError("Contradictions flagged without explanation")
        if normalized and not data["has_contradictions"]:
            data["has_contradictions"] = True
        for claim in data["claims"]:
            if not isinstance(claim, dict):
                raise RuntimeError("Invalid claim")
            if not isinstance(claim.get("text"), str) or not claim["text"].strip():
                raise RuntimeError("Invalid claim text")
            ids = claim.get("source_ids")
            if not isinstance(ids, list) or not ids or not all(
                type(value) is int for value in ids
            ):
                raise RuntimeError("Invalid claim source_ids")
        return data

    @classmethod
    def _parse_planning_response(
        cls,
        response: str,
    ) -> dict[str, list[str]]:
        data = cls._parse_json_response(response)

        queries = data.get("queries")

        if not isinstance(queries, list):
            raise RuntimeError(
                "Retrieval plan must contain queries"
            )

        if len(queries) > 2:
            raise RuntimeError(
                "Too many follow-up queries"
            )

        if not all(
            isinstance(query, str) and query.strip()
            for query in queries
        ):
            raise RuntimeError(
                "Invalid follow-up query"
            )

        return {
            "queries": [
                query.strip()
                for query in queries
            ]
        }

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

    def _plan_follow_up_queries(
        self,
        question: str,
        sources: list[dict],
    ) -> list[str]:
        context = self._build_context(sources)

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

        data = self._parse_planning_response(response)

        return data["queries"]

    def _retrieve_multi_hop(
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
            current_sources = self._build_sources(
                all_results
            )

            if not current_sources:
                break

            follow_up_queries = (
                self._plan_follow_up_queries(
                    question=question,
                    sources=current_sources,
                )
            )

            if not follow_up_queries:
                break

            new_searches = 0

            for query in follow_up_queries:
                normalized_query = (
                    query.casefold().strip()
                )

                if normalized_query in executed_queries:
                    continue

                executed_queries.add(
                    normalized_query
                )

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
            self._build_sources(all_results),
            search_history,
        )

    @staticmethod
    def _validate_claims(data: dict, available_sources: dict[int, dict]) -> None:
        for contradiction in data["contradictions"]:
            for source_id in contradiction["source_ids"]:
                if source_id not in available_sources:
                    raise RuntimeError("LLM cited an unknown contradiction source")
        if not data["is_answerable"]:
            if data["claims"]:
                raise RuntimeError("Unanswerable response must not contain claims")
            return
        if not data["claims"]:
            raise RuntimeError("Answerable response must contain claims")
        for claim in data["claims"]:
            if re.search(r"\[\d+\]", claim["text"]):
                raise RuntimeError("Claim text contains citation markers")
            for source_id in claim["source_ids"]:
                if source_id not in available_sources:
                    raise RuntimeError("LLM cited an unknown source")

    def answer(
        self,
        question: str,
        top_k: int = 8,
        max_hops: int = 3,
    ) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 1 <= max_hops <= 3:
            raise ValueError("max_hops must be between 1 and 3")

        sources, search_history = self._retrieve_multi_hop(
            question=question,
            top_k=top_k,
            max_hops=max_hops,
        )
        if not sources:
            return self._abstention_response(
                question=question,
                search_history=search_history,
            )

        available_sources = {
            source["citation_id"]: source for source in sources
        }
        prompt = (
            f"Question:\n{question}\n\n"
            f"Available evidence:\n{self._build_context(sources)}\n\n"
            "Return the required JSON. Each claim must be grounded in its cited evidence."
        )

        last_error = None
        for attempt in range(2):
            raw_response = self.llm_service.generate(
                prompt=prompt,
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.0,
            )
            try:
                data = self._parse_response(raw_response)
                self._validate_claims(data, available_sources)
                break
            except RuntimeError as error:
                last_error = error
                prompt = (
                    f"{prompt}\n\nYour previous output was invalid: {error}. "
                    "Regenerate the complete JSON with valid claims and existing source IDs."
                )
        else:
            raise RuntimeError(
                "LLM failed to return a valid evidence-grounded response after retry"
            ) from last_error

        contradiction_ids = list(dict.fromkeys(
            source_id
            for contradiction in data["contradictions"]
            for source_id in contradiction["source_ids"]
        ))
        contradiction_details = [
            contradiction["description"] for contradiction in data["contradictions"]
        ]
        if not data["is_answerable"]:
            response = self._abstention_response(
                question=question,
                retrieved_sources=sources,
                search_history=search_history,
            )
            if data["has_contradictions"]:
                response["answer"] = (
                    "The available documents contain conflicting evidence: "
                    + " ".join(contradiction_details)
                )
            response["has_contradictions"] = data["has_contradictions"]
            response["contradictions"] = contradiction_details
            response["sources"] = [available_sources[i] for i in contradiction_ids]
            return response

        answer_parts = []
        cited_ids = []
        for claim in data["claims"]:
            unique_ids = list(dict.fromkeys(claim["source_ids"]))
            for source_id in unique_ids:
                if source_id not in cited_ids:
                    cited_ids.append(source_id)
            citation_text = "".join(f"[{source_id}]" for source_id in unique_ids)
            claim_text = claim["text"].strip()
            answer_parts.append(f"{claim_text} {citation_text}")

        for source_id in contradiction_ids:
            if source_id not in cited_ids:
                cited_ids.append(source_id)
        return {
            "question": question,
            "answer": " ".join(answer_parts),
            "is_answerable": True,
            "has_contradictions": data["has_contradictions"],
            "contradictions": contradiction_details,
            "sources": [available_sources[source_id] for source_id in cited_ids],
            "retrieved_sources": sources,
            "search_history": search_history,
        }

