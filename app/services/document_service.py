from pathlib import Path

from dataset.adapters.json_adapter import load_json_documents
from dataset.adapters.csv_adapter import load_csv_documents
from dataset.adapters.pdf_adapter import load_pdf_documents
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