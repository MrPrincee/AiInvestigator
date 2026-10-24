import csv
from pathlib import Path

from dataset.identifiers import create_document_id
from dataset.models import Document


def load_csv_documents(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format: {path.suffix}")

    documents = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file must contain a header.")

        for row_number, row in enumerate(reader, start=2):
            content = row.get("content") or row.get("text") or ""
            content = content.strip()

            if not content:
                continue

            title = row.get("title") or f"{path.stem} - Row {row_number}"

            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"title", "content", "text"}
            }

            metadata["filename"] = path.name
            metadata["row_number"] = row_number

            document = Document(
                id=create_document_id(title, content),
                title=title,
                content=content,
                source=str(path),
                metadata=metadata,
            )

            documents.append(document)

    return documents