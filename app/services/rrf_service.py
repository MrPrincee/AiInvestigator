def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    fused_results = {}

    for results in result_lists:
        seen_ids = set()

        for rank, result in enumerate(results, start=1):
            result_id = result["chunk_id"]

            if result_id in seen_ids:
                continue

            seen_ids.add(result_id)

            if result_id not in fused_results:
                fused_results[result_id] = {
                    **result,
                    "score": 0.0,
                }

            fused_results[result_id]["score"] += 1 / (k + rank)

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: (-item["score"], item["chunk_id"]),
    )

    return ranked_results[:top_k]