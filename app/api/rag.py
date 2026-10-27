from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.rag_service import RAGService


router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    max_hops: int = Field(default=3, ge=1, le=3)


@router.post("/ask")
def ask_question(
    request: RAGRequest,
    http_request: Request,
):
    service: RAGService = http_request.app.state.rag_service

    try:
        return service.answer(
            question=request.question,
            top_k=request.top_k,
            max_hops=request.max_hops,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error