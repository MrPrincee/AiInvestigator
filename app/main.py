from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.services.hybrid_search_service import HybridSearchService
from app.services.document_crud_service import DocumentCRUDService

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.search_service = HybridSearchService()

    try:
        application.state.crud_service = DocumentCRUDService()

        try:
            yield
        finally:
            application.state.crud_service.close()
    finally:
        application.state.search_service.close()

app = FastAPI(
    title="AI Investigation Engine",
    description="Document ingestion, retrieval and AI-powered investigation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(search_router)


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