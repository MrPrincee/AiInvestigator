from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.llm import router as llm_router
from app.api.rag import router as rag_router
from app.api.search import router as search_router

from app.services.document_crud_service import DocumentCRUDService
from app.services.embedding_service import EmbeddingService
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService
from app.services.qdrant_service import QdrantService
from app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(application: FastAPI):
    qdrant_service = None
    llm_service = None

    try:
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        search_service = HybridSearchService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
        )

        crud_service = DocumentCRUDService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
        )

        llm_service = LLMService()

        application.state.search_service = search_service
        application.state.crud_service = crud_service
        application.state.llm_service = llm_service
        application.state.rag_service = RAGService(
            search_service=search_service,
            llm_service=llm_service,
        )

        yield

    finally:
        if llm_service is not None:
            llm_service.close()

        if qdrant_service is not None:
            qdrant_service.close()


app = FastAPI(
    title="AI Investigation Engine",
    description="Document ingestion, retrieval and AI-powered investigation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(llm_router)
app.include_router(rag_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Investigation Engine",
    }