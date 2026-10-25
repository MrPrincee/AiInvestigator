import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.services.document_service import process_document

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".pdf"}


@router.post("/upload")
def upload_document(file: UploadFile):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format",
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename

            with file_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    output.write(chunk)

            documents = process_document(file_path)

        return {
            "filename": filename,
            "document_count": len(documents),
            "documents": [
                {
                    "id": document.id,
                    "title": document.title,
                    "content": document.content,
                    "source": filename,
                    "metadata": document.metadata,
                }
                for document in documents
            ],
        }

    except (ValueError, UnicodeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    finally:
        file.file.close()