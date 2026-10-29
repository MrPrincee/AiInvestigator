import json
from pathlib import Path
from statistics import mean

from dataset.chunking import fixed_size_chunking, sentence_aware_chunking
from dataset.models import Document


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "hotpotqa_documents.jsonl"
OUTPUT_PATH = ROOT / "data" / "evaluation" / "chunking_results.json"

SAMPLE_SIZE = 1000


def load_documents(limit: int) -> list[Document]:
    documents = []

    with INPUT_PATH.open(encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)

            documents.append(
                Document(
                    id=data["id"],
                    title=data["title"],
                    content=data["content"],
                    source=data["source"],
                    metadata=data.get("metadata", {}),
                )
            )

            if len(documents) >= limit:
                break

    return documents


def evaluate(documents: list[Document], strategy: str) -> dict:
    chunk_lengths = []
    total_chunks = 0
    sentence_endings = 0

    for document in documents:
        if strategy == "fixed_size":
            chunks = fixed_size_chunking(document)
        else:
            chunks = sentence_aware_chunking(document)

        total_chunks += len(chunks)

        for chunk in chunks:
            content = chunk.content.strip()
            chunk_lengths.append(len(content))

            if content.endswith((".", "!", "?")):
                sentence_endings += 1

    return {
        "documents": len(documents),
        "total_chunks": total_chunks,
        "average_chunk_length": round(mean(chunk_lengths), 2),
        "sentence_ending_rate": round(
            sentence_endings / len(chunk_lengths), 4
        ),
    }


def main():
    documents = load_documents(SAMPLE_SIZE)

    results = {
        "sample_size": len(documents),
        "fixed_size": evaluate(documents, "fixed_size"),
        "sentence_aware": evaluate(documents, "sentence_aware"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()