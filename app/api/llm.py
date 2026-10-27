from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.llm_service import LLMService


router = APIRouter(prefix="/llm", tags=["LLM"])


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system_instruction: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    model: str
    response: str


@router.post("/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    http_request: Request,
):
    service: LLMService = http_request.app.state.llm_service

    try:
        response = service.generate(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
        )

        return GenerateResponse(
            model=service.model_name,
            response=response,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error