import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from app.services.qdrant_service import QdrantService
from dataset.identifiers import create_document_id


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hotpotqa_evaluation.jsonl"
)

COLLECTION_NAME = "hotpotqa_50k_hybrid"
QDRANT_URL = "http://localhost:6333"

TARGET_QUESTIONS = 50
QUESTIONS_PER_TYPE = 25


def get_indexed_document_ids(
    qdrant_service: QdrantService,
    collection_name: str,
) -> set[str]:
    client = qdrant_service.client

    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Collection does not exist: {collection_name}"
        )

    document_ids = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            document_id = (point.payload or {}).get("document_id")

            if document_id:
                document_ids.add(document_id)

        if offset is None:
            break

    return document_ids


def get_supporting_documents(example: dict) -> list[dict] | None:
    context = example["context"]
    supporting_facts = example["supporting_facts"]

    required_titles = set(supporting_facts["title"])

    documents_by_title = {}

    for title, sentences in zip(
        context["title"],
        context["sentences"],
        strict=True,
    ):
        content = " ".join(sentences).strip()

        if not content:
            continue

        if title in required_titles:
            documents_by_title[title] = {
                "document_id": create_document_id(title, content),
                "title": title,
            }

    if not required_titles.issubset(documents_by_title):
        return None

    return [
        documents_by_title[title]
        for title in sorted(required_titles)
    ]


def prepare_evaluation_dataset() -> dict:
    print("Connecting to Qdrant...")

    qdrant_service = QdrantService(url=QDRANT_URL)

    try:
        print("Loading indexed document IDs...")

        indexed_document_ids = get_indexed_document_ids(
            qdrant_service=qdrant_service,
            collection_name=COLLECTION_NAME,
        )

    finally:
        qdrant_service.close()

    print(
        f"Indexed unique documents: "
        f"{len(indexed_document_ids):,}"
    )

    print("Loading HotpotQA questions...")

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
    )

    selected = {
        "bridge": [],
        "comparison": [],
    }

    seen_question_ids = set()

    for example in dataset:
        question_type = example["type"]

        if question_type not in selected:
            continue

        if len(selected[question_type]) >= QUESTIONS_PER_TYPE:
            continue

        question_id = example["id"]

        if question_id in seen_question_ids:
            continue

        supporting_documents = get_supporting_documents(example)

        if not supporting_documents:
            continue

        required_document_ids = {
            document["document_id"]
            for document in supporting_documents
        }

        if not required_document_ids.issubset(indexed_document_ids):
            continue

        evaluation_example = {
            "id": question_id,
            "question": example["question"],
            "expected_answer": example["answer"],
            "question_type": question_type,
            "answerable": True,
            "supporting_document_ids": sorted(required_document_ids),
            "supporting_titles": [
                document["title"]
                for document in supporting_documents
            ],
            "supporting_facts": [
                {
                    "title": title,
                    "sentence_id": sentence_id,
                }
                for title, sentence_id in zip(
                    example["supporting_facts"]["title"],
                    example["supporting_facts"]["sent_id"],
                    strict=True,
                )
            ],
        }

        selected[question_type].append(evaluation_example)
        seen_question_ids.add(question_id)

        if all(
            len(questions) >= QUESTIONS_PER_TYPE
            for questions in selected.values()
        ):
            break

    evaluation_questions = (
        selected["bridge"]
        + selected["comparison"]
    )

    if len(evaluation_questions) < TARGET_QUESTIONS:
        raise RuntimeError(
            f"Only {len(evaluation_questions)} eligible questions found. "
            f"Expected {TARGET_QUESTIONS}. "
            "Evaluation dataset was not saved."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for example in evaluation_questions:
            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                ) + "\n"
            )

    statistics = Counter(
        example["question_type"]
        for example in evaluation_questions
    )

    print("\n========== EVALUATION DATASET ==========")
    print(f"Total questions: {len(evaluation_questions)}")
    print(f"Bridge questions: {statistics['bridge']}")
    print(f"Comparison questions: {statistics['comparison']}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Saved to: {OUTPUT_PATH}")

    return {
        "total_questions": len(evaluation_questions),
        "bridge_questions": statistics["bridge"],
        "comparison_questions": statistics["comparison"],
        "output_path": str(OUTPUT_PATH),
    }


if __name__ == "__main__":
    prepare_evaluation_dataset()