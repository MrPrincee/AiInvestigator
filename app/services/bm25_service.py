import math
import re
from collections import Counter


class BM25Service:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")

        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.k1 = k1
        self.b = b
        self.documents = []
        self.document_lengths = []
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.average_document_length = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_lengths = []
        self.term_frequencies = []
        self.document_frequencies = Counter()

        for document in documents:
            tokens = self.tokenize(document["content"])
            frequencies = Counter(tokens)

            self.document_lengths.append(len(tokens))
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())

        total_length = sum(self.document_lengths)

        self.average_document_length = (
            total_length / len(documents)
            if documents
            else 0.0
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self.documents:
            return []

        query_terms = set(self.tokenize(query))
        results = []
        document_count = len(self.documents)

        for index, document in enumerate(self.documents):
            score = 0.0
            document_length = self.document_lengths[index]
            frequencies = self.term_frequencies[index]

            for term in query_terms:
                frequency = frequencies.get(term, 0)

                if frequency == 0:
                    continue

                document_frequency = self.document_frequencies[term]

                idf = math.log(
                    1 + (
                        document_count - document_frequency + 0.5
                    ) / (
                        document_frequency + 0.5
                    )
                )

                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * document_length / self.average_document_length
                )

                score += idf * (
                    frequency * (self.k1 + 1)
                ) / denominator

            if score > 0:
                results.append({
                    **document,
                    "score": score,
                })

        results.sort(key=lambda result: result["score"], reverse=True)

        return results[:top_k]