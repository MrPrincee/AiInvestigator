import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "evaluation"

ANSWERS_PATH = DATA_DIR / "answer_results.jsonl"
DATASET_PATH = DATA_DIR / "hotpotqa_evaluation.jsonl"

MODES = ("standard", "tool_calling")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def extract_citation_ids(answer: str) -> set[int]:
    citation_ids = set()

    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for number in re.findall(r"\d+", group):
            citation_ids.add(int(number))

    return citation_ids


def evaluate_citations(
    answer: dict,
    supporting_document_ids: list[str],
) -> dict:
    gold_ids = set(supporting_document_ids)

    citation_ids = extract_citation_ids(
        answer["predicted_answer"]
    )

    sources_by_citation = {
        int(source["citation_id"]): source
        for source in answer.get("sources", [])
        if source.get("citation_id") is not None
    }

    cited_document_ids = set()
    unresolved_citations = set()

    for citation_id in citation_ids:
        source = sources_by_citation.get(citation_id)

        if source is None or not source.get("document_id"):
            unresolved_citations.add(citation_id)
            continue

        cited_document_ids.add(source["document_id"])

    relevant_ids = cited_document_ids & gold_ids

    precision = (
        len(relevant_ids) / len(cited_document_ids)
        if cited_document_ids
        else 0.0
    )

    recall = (
        len(relevant_ids) / len(gold_ids)
        if gold_ids
        else 0.0
    )

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "id": answer["id"],
        "mode": answer["mode"],
        "citation_precision_proxy": precision,
        "citation_recall": recall,
        "citation_f1_proxy": f1,
        "citation_count": len(citation_ids),
        "cited_document_count": len(cited_document_ids),
        "unresolved_citations": len(unresolved_citations),
        "has_citations": bool(citation_ids),
    }


def print_summary(results: list[dict]) -> None:
    print("\n========== CITATION EVALUATION ==========")

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
        print(f"Questions: {count}")

        for metric in (
            "citation_precision_proxy",
            "citation_recall",
            "citation_f1_proxy",
        ):
            average = sum(
                result[metric]
                for result in mode_results
            ) / count

            print(f"{metric}: {average:.4f}")

        with_citations = sum(
            result["has_citations"]
            for result in mode_results
        )

        unresolved = sum(
            result["unresolved_citations"]
            for result in mode_results
        )

        print(
            f"answers_with_citations: "
            f"{with_citations}/{count}"
        )
        print(f"unresolved_citations: {unresolved}")


def main() -> None:
    answers = load_jsonl(ANSWERS_PATH)
    questions = load_jsonl(DATASET_PATH)

    gold_lookup = {
        question["id"]: question["supporting_document_ids"]
        for question in questions
    }

    answer_lookup = {
        (answer["id"], answer["mode"]): answer
        for answer in answers
    }

    common_ids = set.intersection(
        *[
            {
                question_id
                for question_id, mode in answer_lookup
                if mode == current_mode
                and question_id in gold_lookup
            }
            for current_mode in MODES
        ]
    )

    print(f"Common questions: {len(common_ids)}")

    results = []

    for mode in MODES:
        for question_id in sorted(common_ids):
            answer = answer_lookup[(question_id, mode)]

            result = evaluate_citations(
                answer=answer,
                supporting_document_ids=gold_lookup[question_id],
            )

            results.append(result)

    print_summary(results)

    output_path = DATA_DIR / "citation_results.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()