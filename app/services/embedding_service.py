from sentence_transformers import SentenceTransformer

from dataset.chunking import Chunk


class EmbeddingService:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.vector_size = self.model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts(
            [chunk.content for chunk in chunks]
        )