import uvicorn
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.search import router as search_router

app = FastAPI(
    title="AI Investigation Engine",
    description="Document ingestion, retrieval and AI-powered investigation API",
    version="0.1.0",
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
        reload=True
    )