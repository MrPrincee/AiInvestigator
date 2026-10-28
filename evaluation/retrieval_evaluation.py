import json
from pathlib import Path

from app.services.hybrid_search_service import HybridSearchService


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hotpotqa_evaluation.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_results.json"
)

COLLECTION_NAME = "hotpotqa_50k_hybrid"
SEARCH_MODES = ("semantic", "keyword", "hybrid")
K_VALUES = (1, 3, 5, 10)


def load_evaluation_questions() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def calculate_recall(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def calculate_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    for rank, document_id in enumerate(retrieved_ids[:k], start=1):
        if document_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval() -> dict:
    questions = load_evaluation_questions()

    if not questions:
        raise ValueError("Evaluation dataset is empty")

    search_service = HybridSearchService(
        collection_name=COLLECTION_NAME,
    )

    results = {}

    try:
        for mode in SEARCH_MODES:
            print(f"\nEvaluating {mode} search...", flush=True)

            metrics = {
                f"recall@{k}": 0.0
                for k in K_VALUES
            }

            metrics.update({
                f"mrr@{k}": 0.0
                for k in K_VALUES
            })

            question_results = []

            for index, example in enumerate(questions, start=1):
                relevant_ids = set(
                    example["supporting_document_ids"]
                )

                retrieved_chunks = search_service.search(
                    query=example["question"],
                    mode=mode,
                    top_k=max(K_VALUES),
                )

                retrieved_ids = []
                seen_document_ids = set()

                for chunk in retrieved_chunks:
                    document_id = chunk.get("document_id")

                    if (
                        not document_id
                        or document_id in seen_document_ids
                    ):
                        continue

                    retrieved_ids.append(document_id)
                    seen_document_ids.add(document_id)

                question_metrics = {}

                for k in K_VALUES:
                    recall = calculate_recall(
                        retrieved_ids,
                        relevant_ids,
                        k,
                    )

                    reciprocal_rank = calculate_reciprocal_rank(
                        retrieved_ids,
                        relevant_ids,
                        k,
                    )

                    metrics[f"recall@{k}"] += recall
                    metrics[f"mrr@{k}"] += reciprocal_rank

                    question_metrics[f"recall@{k}"] = recall
                    question_metrics[f"mrr@{k}"] = reciprocal_rank

                question_results.append({
                    "id": example["id"],
                    "question": example["question"],
                    "question_type": example["question_type"],
                    "relevant_document_ids": sorted(relevant_ids),
                    "retrieved_document_ids": retrieved_ids,
                    "metrics": question_metrics,
                })

                if index % 10 == 0:
                    print(
                        f"Processed {index}/{len(questions)} questions",
                        flush=True,
                    )

            for metric_name in metrics:
                metrics[metric_name] /= len(questions)

            results[mode] = {
                "question_count": len(questions),
                "metrics": metrics,
                "questions": question_results,
            }

    finally:
        search_service.close()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========== RETRIEVAL EVALUATION ==========")

    for mode, result in results.items():
        print(f"\n{mode.upper()}")

        for metric_name, value in result["metrics"].items():
            print(f"{metric_name}: {value:.4f}")

    print(f"\nSaved to: {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    evaluate_retrieval()