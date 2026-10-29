import re


def validate_claims(
    data: dict,
    available_sources: dict[int, dict],
) -> None:
    if data["has_contradictions"] and data["is_answerable"]:
        raise RuntimeError(
            "Conflicting evidence cannot produce an answerable response"
        )

    for contradiction in data["contradictions"]:
        for source_id in contradiction["source_ids"]:
            if source_id not in available_sources:
                raise RuntimeError(
                    "LLM cited an unknown contradiction source"
                )

    if not data["is_answerable"]:
        if data["claims"]:
            raise RuntimeError(
                "Unanswerable response must not contain claims"
            )
        return

    if not data["claims"]:
        raise RuntimeError(
            "Answerable response must contain claims"
        )

    for claim in data["claims"]:
        if re.search(r"\[\d+\]", claim["text"]):
            raise RuntimeError(
                "Claim text contains citation markers"
            )

        for source_id in claim["source_ids"]:
            if source_id not in available_sources:
                raise RuntimeError(
                    "LLM cited an unknown source"
                )