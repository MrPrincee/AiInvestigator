import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from app.schemas.documents import (
    DocumentCreateRequest,
    DocumentListResponse,
    DocumentMutationResponse,
    DocumentResponse,
    DocumentUpdateRequest,
)
from app.services.document_service import process_document
from app.services.document_crud_service import DocumentCRUDService


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".pdf"}


def get_crud_service(request: Request) -> DocumentCRUDService:
    return request.app.state.crud_service


def handle_crud_error(error: Exception) -> None:
    if isinstance(error, LookupError):
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    if isinstance(error, ValueError):
        detail = str(error)

        if "already exists" in detail:
            raise HTTPException(
                status_code=409,
                detail=detail,
            ) from error

        raise HTTPException(
            status_code=400,
            detail=detail,
        ) from error

    raise error


@router.post(
    "",
    response_model=DocumentMutationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    data: DocumentCreateRequest,
    request: Request,
):
    service = get_crud_service(request)

    try:
        return service.create_document(
            title=data.title,
            content=data.content,
            source=data.source,
            metadata=data.metadata,
            chunking_strategy=data.chunking_strategy,
        )
    except (ValueError, LookupError) as error:
        handle_crud_error(error)


@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents(request: Request):
    service = get_crud_service(request)

    return service.list_documents()


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: str,
    request: Request,
):
    service = get_crud_service(request)

    try:
        document = service.get_document(document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        return document

    except ValueError as error:
        handle_crud_error(error)


@router.put(
    "/{document_id}",
    response_model=DocumentMutationResponse,
)
def update_document(
    document_id: str,
    data: DocumentUpdateRequest,
    request: Request,
):
    service = get_crud_service(request)

    try:
        return service.update_document(
            document_id=document_id,
            title=data.title,
            content=data.content,
            metadata=data.metadata,
            chunking_strategy=data.chunking_strategy,
        )
    except (ValueError, LookupError) as error:
        handle_crud_error(error)


@router.delete(
    "/{document_id}",
    response_model=DocumentMutationResponse,
)
def delete_document(
    document_id: str,
    request: Request,
):
    service = get_crud_service(request)

    try:
        return service.delete_document(document_id)
    except (ValueError, LookupError) as error:
        handle_crud_error(error)


@router.post("/upload")
def upload_document(
    file: UploadFile,
    request: Request,
    chunking_strategy: str = "fixed_size",
):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format",
        )

    if chunking_strategy not in {"fixed_size", "sentence_aware"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported chunking strategy",
        )

    service = get_crud_service(request)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / filename

            with file_path.open("wb") as output:
                while chunk := file.file.read(1024 * 1024):
                    output.write(chunk)

            documents = process_document(file_path)

        results = []
        duplicates = []

        for document in documents:
            try:
                result = service.create_document(
                    title=document.title,
                    content=document.content,
                    source=filename,
                    metadata=document.metadata,
                    chunking_strategy=chunking_strategy,
                )

                results.append(result)

            except ValueError as error:
                if "already exists" in str(error):
                    duplicates.append({
                        "title": document.title,
                        "reason": str(error),
                    })
                    continue

                raise

        return {
            "filename": filename,
            "document_count": len(documents),
            "indexed_count": len(results),
            "duplicate_count": len(duplicates),
            "documents": results,
            "duplicates": duplicates,
        }

    except (ValueError, UnicodeError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    finally:
        file.file.close()