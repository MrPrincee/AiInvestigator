from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.llm import router as llm_router
from app.api.rag import router as rag_router
from app.api.search import router as search_router
from app.services.document_crud_service import DocumentCRUDService
from app.services.hybrid_search_service import HybridSearchService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(application: FastAPI):
    search_service = None
    crud_service = None
    llm_service = None

    try:
        search_service = HybridSearchService()
        crud_service = DocumentCRUDService()
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

        if crud_service is not None:
            crud_service.close()

        if search_service is not None:
            search_service.close()


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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )