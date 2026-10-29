import json
import re
from typing import Any


def parse_json_response(response: str) -> dict[str, Any]:
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
        raise RuntimeError("LLM returned invalid JSON") from error

    if not isinstance(data, dict):
        raise RuntimeError("LLM response must be a JSON object")

    return data


def parse_response(response: str) -> dict[str, Any]:
    data = parse_json_response(response)

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
            normalized.append(
                {
                    "description": item.strip(),
                    "source_ids": [],
                }
            )

        elif isinstance(item, dict):
            description = next(
                (
                    item[key].strip()
                    for key in (
                        "description",
                        "explanation",
                        "conflict",
                        "text",
                        "details",
                    )
                    if isinstance(item.get(key), str)
                    and item[key].strip()
                ),
                None,
            )

            if description is None:
                parts = [
                    f"{key}: {value}"
                    for key, value in item.items()
                    if key not in (
                        "source_ids",
                        "sources",
                        "citation_ids",
                    )
                ]

                description = "; ".join(parts).strip()

            ids = item.get(
                "source_ids",
                item.get("citation_ids", []),
            )

            if not isinstance(ids, list) or not all(
                type(value) is int for value in ids
            ):
                raise RuntimeError(
                    "Invalid contradiction source_ids"
                )

            if not description:
                raise RuntimeError(
                    "Invalid contradiction description"
                )

            normalized.append(
                {
                    "description": description,
                    "source_ids": list(dict.fromkeys(ids)),
                }
            )

        else:
            raise RuntimeError("Invalid contradiction entry")

    data["contradictions"] = normalized

    if data["has_contradictions"] and not normalized:
        raise RuntimeError(
            "Contradictions flagged without explanation"
        )

    if normalized and not data["has_contradictions"]:
        data["has_contradictions"] = True

    for claim in data["claims"]:
        if not isinstance(claim, dict):
            raise RuntimeError("Invalid claim")

        if (
            not isinstance(claim.get("text"), str)
            or not claim["text"].strip()
        ):
            raise RuntimeError("Invalid claim text")

        ids = claim.get("source_ids")

        if not isinstance(ids, list) or not ids or not all(
            type(value) is int for value in ids
        ):
            raise RuntimeError("Invalid claim source_ids")

    return data


def parse_planning_response(
    response: str,
) -> dict[str, list[str]]:
    data = parse_json_response(response)

    queries = data.get("queries")

    if not isinstance(queries, list):
        raise RuntimeError(
            "Retrieval plan must contain queries"
        )

    if len(queries) > 2:
        raise RuntimeError("Too many follow-up queries")

    if not all(
        isinstance(query, str) and query.strip()
        for query in queries
    ):
        raise RuntimeError("Invalid follow-up query")

    return {
        "queries": [
            query.strip()
            for query in queries
        ]
    }