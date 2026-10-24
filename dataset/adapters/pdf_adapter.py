from pathlib import Path

import pymupdf

from dataset.identifiers import create_document_id
from dataset.models import Document


def load_pdf_documents(file_path: str | Path) -> list[Document]:
    path = Path(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file format: {path.suffix}")

    documents = []

    with pymupdf.open(path) as pdf:
        pdf_metadata = pdf.metadata or {}

        for page_number, page in enumerate(pdf, start=1):
            content = page.get_text("text").strip()

            if not content:
                continue

            title = pdf_metadata.get("title") or path.stem
            page_title = f"{title} - Page {page_number}"

            document = Document(
                id=create_document_id(page_title, content),
                title=page_title,
                content=content,
                source=str(path),
                metadata={
                    "filename": path.name,
                    "page_number": page_number,
                    "total_pages": len(pdf),
                    "author": pdf_metadata.get("author"),
                    "creation_date": pdf_metadata.get("creationDate"),
                    "file_format": "pdf",
                },
            )

            documents.append(document)

    return documents