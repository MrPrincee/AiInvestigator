import json
from pathlib import Path

from dataset.identifiers import create_document_id
from dataset.models import Document


def load_json_documents(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    documents = []

    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]
        else:
            data = json.load(file)
            records = data if isinstance(data, list) else [data]

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each document must be a JSON object.")

        content = record.get("content") or record.get("text") or ""

        if not isinstance(content, str):
            raise ValueError("Document content must be a string.")

        content = content.strip()

        if not content:
            continue

        title = record.get("title") or path.stem

        if not isinstance(title, str):
            raise ValueError("Document title must be a string.")

        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"id", "title", "content", "text", "source", "metadata"}
        }

        if isinstance(record.get("metadata"), dict):
            metadata.update(record["metadata"])

        metadata["filename"] = path.name

        document = Document(
            id=create_document_id(title, content),
            title=title,
            content=content,
            source=str(path),
            metadata=metadata,
        )

        documents.append(document)

    return documents