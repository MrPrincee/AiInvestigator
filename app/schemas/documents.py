from typing import Any, Literal

from pydantic import BaseModel, Field


ChunkingStrategy = Literal["fixed_size", "sentence_aware"]


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: ChunkingStrategy = "fixed_size"


class DocumentUpdateRequest(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: ChunkingStrategy = "fixed_size"


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentSummary]


class DocumentMutationResponse(BaseModel):
    status: str
    document_id: str
    chunks_affected: int