import json
import re

from app.services.llm_service import LLMService


class AnswerCorrectnessEvaluator:
    SYSTEM_INSTRUCTION = """
You are an impartial evaluator of question-answering systems.

Evaluate whether the predicted answer correctly answers the question,
using the expected answer as the reference.

Rules:
- A correct answer may be part of a longer sentence.
- Ignore citation markers such as [1] and [2].
- Do not require exact wording.
- Accept semantically equivalent answers.
- Do not reward an answer merely for mentioning the expected answer
  if it ultimately gives a different or contradictory conclusion.
- An abstention is incorrect when the reference contains an answer.
- If the question or reference answer is ambiguous or inconsistent,
  mark the case for review.
- Do not use external knowledge.
- Return only valid JSON.

Return:
{
    "score": 0.0,
    "reason": "Short explanation",
    "needs_review": false
}

Scoring:
1.0 = Correct answer
0.5 = Partially correct or incomplete answer
0.0 = Incorrect or missing answer
"""

    def __init__(self):
        self.llm_service = LLMService()

    @staticmethod
    def parse_response(response: str) -> dict:
        cleaned = response.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(
                r"^```(?:json)?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        if not isinstance(data, dict):
            raise ValueError("Evaluator response must be an object")

        score = data.get("score")
        reason = data.get("reason")
        needs_review = data.get("needs_review")

        if type(score) not in (int, float) or score not in (0, 0.5, 1):
            raise ValueError("Invalid correctness score")

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalid evaluation reason")

        if type(needs_review) is not bool:
            raise ValueError("Invalid needs_review value")

        return {
            "score": float(score),
            "reason": reason.strip(),
            "needs_review": needs_review,
        }

    def evaluate(
        self,
        question: str,
        expected_answer: str,
        predicted_answer: str,
    ) -> dict:
        prompt = (
            f"Question:\n{question}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Predicted answer:\n{predicted_answer}\n\n"
            "Evaluate the predicted answer and return the required JSON."
        )

        response = self.llm_service.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            temperature=0.0,
        )

        return self.parse_response(response)

    def close(self):
        self.llm_service.close()