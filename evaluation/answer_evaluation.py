import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "hotpotqa_evaluation.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "answer_results.jsonl"
)

API_URL = "http://127.0.0.1:8000/rag/ask"

MODES = ("standard", "tool_calling")
MAX_RETRIES = 3
RETRY_DELAY = 10


def normalize_answer(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = text.lower()

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def exact_match(prediction: str, expected: str) -> float:
    return float(
        normalize_answer(prediction)
        == normalize_answer(expected)
    )


def token_f1(prediction: str, expected: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    expected_tokens = normalize_answer(expected).split()

    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)

    common = Counter(predicted_tokens) & Counter(expected_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)

    return 2 * precision * recall / (precision + recall)


def citation_coverage(
    sources: list[dict],
    supporting_document_ids: list[str],
) -> float:
    expected = set(supporting_document_ids)

    if not expected:
        return 0.0

    cited = {
        source["document_id"]
        for source in sources
        if source.get("document_id")
    }

    return len(cited & expected) / len(expected)


def load_questions() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_existing_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_result(result: dict) -> None:
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


def ask_rag(question: str, mode: str) -> dict:
    payload = {
        "question": question,
        "top_k": 8,
        "max_hops": 3,
        "retrieval_mode": mode,
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=240) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                details = error.read().decode("utf-8")
                raise RuntimeError(
                    f"API returned {error.code}: {details}"
                ) from error

            if attempt == MAX_RETRIES:
                raise

            print(
                f"API error {error.code}. "
                f"Retrying in {RETRY_DELAY} seconds...",
                flush=True,
            )

        except (URLError, TimeoutError):
            if attempt == MAX_RETRIES:
                raise

            print(
                f"Connection error. "
                f"Retrying in {RETRY_DELAY} seconds...",
                flush=True,
            )

        time.sleep(RETRY_DELAY)

    raise RuntimeError("RAG request failed")


def evaluate_answer(example: dict, mode: str) -> dict:
    start_time = time.perf_counter()

    response = ask_rag(
        question=example["question"],
        mode=mode,
    )

    elapsed = time.perf_counter() - start_time

    answer = response["answer"]
    expected = example["expected_answer"]
    sources = response.get("sources", [])

    return {
        "id": example["id"],
        "mode": mode,
        "question_type": example["question_type"],
        "question": example["question"],
        "expected_answer": expected,
        "predicted_answer": answer,
        "is_answerable": response["is_answerable"],
        "has_contradictions": response["has_contradictions"],
        "exact_match": exact_match(answer, expected),
        "token_f1": token_f1(answer, expected),
        "citation_coverage": citation_coverage(
            sources,
            example["supporting_document_ids"],
        ),
        "sources": sources,
        "retrieved_sources": response.get("retrieved_sources", []),
        "search_history": response.get("search_history", []),
        "elapsed_seconds": round(elapsed, 2),
    }


def print_summary(results: list[dict]) -> None:
    print("\n========== ANSWER EVALUATION ==========")

    for mode in MODES:
        mode_results = [
            result
            for result in results
            if result["mode"] == mode
        ]

        if not mode_results:
            continue

        count = len(mode_results)

        print(f"\n{mode.upper()}")
        print(f"Questions evaluated: {count}")

        for metric in (
            "exact_match",
            "token_f1",
            "citation_coverage",
        ):
            average = sum(
                result[metric]
                for result in mode_results
            ) / count

            print(f"{metric}: {average:.4f}")

        abstentions = sum(
            not result["is_answerable"]
            for result in mode_results
        )

        print(f"abstention_rate: {abstentions / count:.4f}")

        average_time = sum(
            result["elapsed_seconds"]
            for result in mode_results
        ) / count

        print(f"average_seconds: {average_time:.2f}")


def main(limit: int) -> None:
    questions = load_questions()

    if not 1 <= limit <= len(questions):
        raise ValueError(
            f"Limit must be between 1 and {len(questions)}"
        )

    existing_results = load_existing_results()

    completed = {
        (result["id"], result["mode"])
        for result in existing_results
    }

    for mode in MODES:
        print(f"\nEvaluating {mode}...", flush=True)

        for index, example in enumerate(
            questions[:limit],
            start=1,
        ):
            key = (example["id"], mode)

            if key in completed:
                print(
                    f"[{index}/{limit}] Already completed",
                    flush=True,
                )
                continue

            print(
                f"[{index}/{limit}] {example['question']}",
                flush=True,
            )

            try:
                result = evaluate_answer(example, mode)

            except Exception as error:
                print(
                    f"Evaluation stopped: {error}",
                    flush=True,
                )
                raise

            save_result(result)

            existing_results.append(result)
            completed.add(key)

            print(
                f"EM={result['exact_match']:.2f}, "
                f"F1={result['token_f1']:.2f}, "
                f"Citations={result['citation_coverage']:.2f}",
                flush=True,
            )

    print_summary(existing_results)

    print(f"\nSaved to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    arguments = parser.parse_args()

    main(limit=arguments.limit)