import json
from collections.abc import Iterator
from pathlib import Path

from dataset.adapters.csv_adapter import load_csv_documents
from dataset.adapters.json_adapter import load_json_documents
from dataset.adapters.pdf_adapter import load_pdf_documents
from dataset.chunking import Chunk, fixed_size_chunking, sentence_aware_chunking
from dataset.models import Document


def process_document(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in {".json", ".jsonl"}:
        return load_json_documents(path)

    if extension == ".csv":
        return load_csv_documents(path)

    if extension == ".pdf":
        return load_pdf_documents(path)

    raise ValueError(f"Unsupported file format: {extension}")


def chunk_documents(
    documents: list[Document],
    strategy: str = "fixed_size",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    chunks = []

    for document in documents:
        if strategy == "fixed_size":
            chunks.extend(
                fixed_size_chunking(
                    document,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
        elif strategy == "sentence_aware":
            chunks.extend(
                sentence_aware_chunking(
                    document,
                    chunk_size=chunk_size,
                )
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    return chunks


def read_document_batches(
    file_path: str | Path,
    batch_size: int = 32,
) -> Iterator[list[Document]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    batch = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            data = json.loads(line)
            batch.append(Document(**data))

            if len(batch) == batch_size:
                yield batch
                batch = []

    if batch:
        yield batch