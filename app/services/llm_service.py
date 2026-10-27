import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


class LLMService:
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        load_dotenv()

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        )

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=resolved_api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise RuntimeError("LLM returned an empty response")

        return response.text

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[Callable[..., Any]],
        system_instruction: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if not tools:
            raise ValueError("At least one tool is required")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=5,
            ),
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise RuntimeError("LLM returned an empty response")

        return response.text

    def close(self) -> None:
        self.client.close()