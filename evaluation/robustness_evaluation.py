import json
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000"

UNANSWERABLE_TESTS = [
    {
        "question": "What is the private bank account PIN of the Oberoi family?",
        "expected_answerable": False,
    },
    {
        "question": "What is the personal email password of the founder of The Oberoi Group?",
        "expected_answerable": False,
    },
    {
        "question": "What was the exact temperature inside the Oberoi Group headquarters at 09:17 on January 5, 1998?",
        "expected_answerable": False,
    },
    {
        "question": "What is the private Wi-Fi password of the Oberoi family residence?",
        "expected_answerable": False,
    },
    {
        "question": "What was the exact number of raindrops that fell on the Oberoi Group headquarters on June 1, 2001?",
        "expected_answerable": False,
    },
]


CONTRADICTION_TESTS = [
    {
        "name": "Project launch year",
        "question": "According to the investigation records, in what year was Project Atlas launched?",
        "documents": [
            (
                "Atlas launch record A",
                "According to the investigation records, Project Atlas was launched in 2022.",
            ),
            (
                "Atlas launch record B",
                "According to the investigation records, Project Atlas was launched in 2024.",
            ),
        ],
    },
    {
        "name": "Company headquarters",
        "question": "According to the investigation records, where is Meridian Labs headquartered?",
        "documents": [
            (
                "Meridian headquarters record A",
                "According to the investigation records, Meridian Labs is headquartered in Berlin.",
            ),
            (
                "Meridian headquarters record B",
                "According to the investigation records, Meridian Labs is headquartered in Paris.",
            ),
        ],
    },
    {
        "name": "Project budget",
        "question": "According to the investigation records, what is the total budget of Project Orion?",
        "documents": [
            (
                "Orion budget record A",
                "According to the investigation records, the total budget of Project Orion is 5 million dollars.",
            ),
            (
                "Orion budget record B",
                "According to the investigation records, the total budget of Project Orion is 8 million dollars.",
            ),
        ],
    },
    {
        "name": "Product release date",
        "question": "According to the investigation records, when was the Helios platform released?",
        "documents": [
            (
                "Helios release record A",
                "According to the investigation records, the Helios platform was released on March 10, 2023.",
            ),
            (
                "Helios release record B",
                "According to the investigation records, the Helios platform was released on June 15, 2023.",
            ),
        ],
    },
    {
        "name": "CEO identity",
        "question": "According to the investigation records, who is the CEO of Northstar Technologies?",
        "documents": [
            (
                "Northstar CEO record A",
                "According to the investigation records, the CEO of Northstar Technologies is Alice Morgan.",
            ),
            (
                "Northstar CEO record B",
                "According to the investigation records, the CEO of Northstar Technologies is David Carter.",
            ),
        ],
    },
]


def api_request(
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    data = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    request = Request(
        f"{API_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )

    with urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode("utf-8"))


def ask_rag(question: str) -> dict:
    return api_request(
        "POST",
        "/rag/ask",
        {
            "question": question,
            "top_k": 8,
            "max_hops": 2,
            "retrieval_mode": "standard",
        },
    )


def evaluate_unanswerable() -> None:
    passed = 0
    completed = 0

    print("\n========== UNANSWERABLE EVALUATION ==========")

    for index, test in enumerate(UNANSWERABLE_TESTS, start=1):
        print(
            f"\n[{index}/{len(UNANSWERABLE_TESTS)}] "
            f"{test['question']}",
            flush=True,
        )

        try:
            response = ask_rag(test["question"])

            actual = response["is_answerable"]
            expected = test["expected_answerable"]

            success = actual == expected

            completed += 1
            passed += int(success)

            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Result: {'PASS' if success else 'FAIL'}")
            print(f"Answer: {response['answer']}")

        except Exception as error:
            print(f"ERROR: {error}", flush=True)

    print("\n========== UNANSWERABLE SUMMARY ==========")
    print(f"Passed: {passed}/{len(UNANSWERABLE_TESTS)}")
    print(f"Completed: {completed}/{len(UNANSWERABLE_TESTS)}")
    print(
        f"Accuracy: "
        f"{passed / len(UNANSWERABLE_TESTS):.2%}"
    )


def evaluate_contradictions() -> None:
    passed = 0
    completed = 0

    print("\n========== CONTRADICTION EVALUATION ==========")

    for index, test in enumerate(CONTRADICTION_TESTS, start=1):
        print(
            f"\n[{index}/{len(CONTRADICTION_TESTS)}] "
            f"{test['name']}",
            flush=True,
        )

        document_ids = []

        try:
            for title, content in test["documents"]:
                result = api_request(
                    "POST",
                    "/documents",
                    {
                        "title": title,
                        "content": content,
                        "source": "robustness_evaluation",
                        "metadata": {
                            "evaluation": "contradiction",
                        },
                        "chunking_strategy": "fixed_size",
                    },
                )

                document_ids.append(result["document_id"])

            response = api_request(
                "POST",
                "/rag/ask",
                {
                    "question": test["question"],
                    "top_k": 12,
                    "max_hops": 2,
                    "retrieval_mode": "standard",
                },
            )

            retrieved_ids = {
                source["document_id"]
                for source in response.get("retrieved_sources", [])
            }

            both_retrieved = set(document_ids).issubset(
                retrieved_ids
            )

            detected = (
                response["has_contradictions"] is True
                and response["is_answerable"] is False
                and bool(response.get("contradictions"))
            )

            success = both_retrieved and detected

            completed += 1
            passed += int(success)

            print(f"Both documents retrieved: {both_retrieved}")
            print(
                f"Has contradictions: "
                f"{response['has_contradictions']}"
            )
            print(
                f"Is answerable: "
                f"{response['is_answerable']}"
            )
            print(
                f"Contradictions: "
                f"{response.get('contradictions', [])}"
            )
            print(f"Result: {'PASS' if success else 'FAIL'}")

        except Exception as error:
            print(f"ERROR: {error}", flush=True)

        finally:
            for document_id in document_ids:
                try:
                    api_request(
                        "DELETE",
                        f"/documents/{document_id}",
                    )
                except Exception as error:
                    print(
                        f"Cleanup failed for {document_id}: "
                        f"{error}"
                    )

    print("\n========== CONTRADICTION SUMMARY ==========")
    print(f"Passed: {passed}/{len(CONTRADICTION_TESTS)}")
    print(f"Completed: {completed}/{len(CONTRADICTION_TESTS)}")
    print(
        f"Accuracy: "
        f"{passed / len(CONTRADICTION_TESTS):.2%}"
    )


def main() -> None:
    evaluate_unanswerable()
    evaluate_contradictions()


if __name__ == "__main__":
    main()