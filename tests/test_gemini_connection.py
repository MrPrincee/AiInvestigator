from app.services.llm_service import LLMService


def test_gemini_connection():
    llm = LLMService()

    try:
        response = llm.generate(
            prompt="What is the capital of Georgia? Answer in one sentence."
        )

        print(f"\nGemini response: {response}")

        assert isinstance(response, str)
        assert response.strip()

    finally:
        llm.close()