import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "evaluation"

ANSWERS_PATH = DATA_DIR / "answer_results.jsonl"
CORRECTNESS_PATH = DATA_DIR / "correctness_results.jsonl"

MODES = ("standard", "tool_calling")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main() -> None:
    answers = load_jsonl(ANSWERS_PATH)
    correctness = load_jsonl(CORRECTNESS_PATH)

    answer_lookup = {
        (item["id"], item["mode"]): item
        for item in answers
    }

    correctness_lookup = {
        (item["id"], item["mode"]): item
        for item in correctness
    }

    common_ids = set.intersection(
        *[
            {
                question_id
                for question_id, mode in answer_lookup
                if mode == current_mode
                and (question_id, mode) in correctness_lookup
            }
            for current_mode in MODES
        ]
    )

    print(f"\nCommon evaluated questions: {len(common_ids)}")

    for mode in MODES:
        print(f"\n========== {mode.upper()} ==========")

        evaluated = []
        errors = []
        review = []
        low_coverage = []

        for question_id in sorted(common_ids):
            key = (question_id, mode)

            answer = answer_lookup[key]
            judgment = correctness_lookup[key]

            score = judgment["answer_correctness"]

            evaluated.append(score)

            if score < 1:
                errors.append((answer, judgment))

            if judgment.get("needs_review", False):
                review.append((answer, judgment))

            if answer["citation_coverage"] < 1:
                low_coverage.append(answer)

        if not evaluated:
            print("No evaluated questions.")
            continue

        average = sum(evaluated) / len(evaluated)

        print(f"Questions: {len(evaluated)}")
        print(f"Answer Correctness: {average:.4f}")
        print(f"Incorrect/Partial: {len(errors)}")
        print(f"Needs Review: {len(review)}")
        print(f"Incomplete Citation Coverage: {len(low_coverage)}")

        print("\n--- INCORRECT OR PARTIAL ANSWERS ---")

        for answer, judgment in errors:
            print(f"\nQuestion: {answer['question']}")
            print(f"Expected: {answer['expected_answer']}")
            print(f"Predicted: {answer['predicted_answer']}")
            print(f"Score: {judgment['answer_correctness']}")
            print(f"Reason: {judgment.get('reason', '')}")
            print(f"Citation Coverage: {answer['citation_coverage']}")

        print("\n--- NEEDS MANUAL REVIEW ---")

        for answer, judgment in review:
            print(f"\nQuestion: {answer['question']}")
            print(f"Expected: {answer['expected_answer']}")
            print(f"Predicted: {answer['predicted_answer']}")
            print(f"Reason: {judgment.get('reason', '')}")

    print("\n========== MODE COMPARISON ==========")

    standard_wins = 0
    tool_wins = 0
    ties = 0

    for question_id in common_ids:
        standard = correctness_lookup[
            (question_id, "standard")
        ]["answer_correctness"]

        tool = correctness_lookup[
            (question_id, "tool_calling")
        ]["answer_correctness"]

        if standard > tool:
            standard_wins += 1
        elif tool > standard:
            tool_wins += 1
        else:
            ties += 1

    print(f"Standard wins: {standard_wins}")
    print(f"Tool Calling wins: {tool_wins}")
    print(f"Ties: {ties}")


if __name__ == "__main__":
    main()