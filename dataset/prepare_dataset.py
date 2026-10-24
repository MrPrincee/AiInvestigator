import json
from dataclasses import asdict
from pathlib import Path

from datasets import load_dataset

from dataset.adapters.hotpotqa_adapter import convert_context_to_documents
from dataset.deduplication import deduplicate_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hotpotqa_documents.jsonl"


def main():
    print("Loading HotpotQA dataset...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
    )

    total_documents = 0

    def generate_documents():
        nonlocal total_documents

        for example in dataset:
            for source_document in convert_context_to_documents(example):
                total_documents += 1
                yield source_document

    unique_documents = deduplicate_documents(generate_documents())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    unique_count = 0

    print("Preparing and saving unique documents...")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for document in unique_documents:
            file.write(
                json.dumps(asdict(document), ensure_ascii=False) + "\n"
            )
            unique_count += 1

    duplicate_count = total_documents - unique_count

    print("\n========== DATASET STATISTICS ==========")
    print(f"Questions: {len(dataset):,}")
    print(f"Total document occurrences: {total_documents:,}")
    print(f"Unique documents: {unique_count:,}")
    print(f"Duplicate occurrences: {duplicate_count:,}")

    if total_documents > 0:
        duplicate_percentage = duplicate_count / total_documents * 100
        print(f"Duplicate percentage: {duplicate_percentage:.2f}%")

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()