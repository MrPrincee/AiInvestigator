from fastembed import TextEmbedding

from dataset.chunking import Chunk


class EmbeddingService:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = TextEmbedding(model_name=self.MODEL_NAME)
        self.vector_size = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.embed(
            texts,
            batch_size=32,
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.embed_texts(
            [chunk.content for chunk in chunks]
        )