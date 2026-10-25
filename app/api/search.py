from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.hybrid_search_service import HybridSearchService


router = APIRouter(tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=100)


@router.post("/search")
def search_documents(request: SearchRequest):
    service = None

    try:
        service = HybridSearchService()

        results = service.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "mode": request.mode,
            "total_results": len(results),
            "results": results,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    finally:
        if service is not None:
            service.close()