import json
import time
from pathlib import Path

from evaluation.answer_correctness import AnswerCorrectnessEvaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "answer_results.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "correctness_results.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def save_result(result: dict) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


def main():
    answers = load_jsonl(RESULTS_PATH)

    existing = (
        load_jsonl(OUTPUT_PATH)
        if OUTPUT_PATH.exists()
        else []
    )

    completed = {
        (result["id"], result["mode"])
        for result in existing
    }

    evaluator = AnswerCorrectnessEvaluator()

    try:
        for index, answer in enumerate(answers, start=1):
            key = (answer["id"], answer["mode"])

            if key in completed:
                print(
                    f"[{index}/{len(answers)}] Already evaluated",
                    flush=True,
                )
                continue

            print(
                f"[{index}/{len(answers)}] "
                f"{answer['mode']}: {answer['question']}",
                flush=True,
            )

            for attempt in range(3):
                try:
                    evaluation = evaluator.evaluate(
                        question=answer["question"],
                        expected_answer=answer["expected_answer"],
                        predicted_answer=answer["predicted_answer"],
                    )
                    break

                except Exception as error:
                    if attempt == 2:
                        raise

                    print(
                        f"Evaluation failed: {error}. Retrying...",
                        flush=True,
                    )
                    time.sleep(10)

            result = {
                "id": answer["id"],
                "mode": answer["mode"],
                "question": answer["question"],
                "expected_answer": answer["expected_answer"],
                "predicted_answer": answer["predicted_answer"],
                "answer_correctness": evaluation["score"],
                "reason": evaluation["reason"],
                "needs_review": evaluation["needs_review"],
            }

            save_result(result)
            existing.append(result)
            completed.add(key)

            print(
                f"Correctness={result['answer_correctness']:.2f}, "
                f"Needs review={result['needs_review']}",
                flush=True,
            )

    finally:
        evaluator.close()

    print("\n========== ANSWER CORRECTNESS ==========")

    for mode in ("standard", "tool_calling"):
        mode_results = [
            result
            for result in existing
            if result["mode"] == mode
        ]

        if not mode_results:
            continue

        average = sum(
            result["answer_correctness"]
            for result in mode_results
        ) / len(mode_results)

        reviews = sum(
            result["needs_review"]
            for result in mode_results
        )

        print(f"\n{mode.upper()}")
        print(f"Questions: {len(mode_results)}")
        print(f"Answer Correctness: {average:.4f}")
        print(f"Needs Review: {reviews}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()