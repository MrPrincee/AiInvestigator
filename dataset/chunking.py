import re
from dataclasses import dataclass, field
from typing import Any

from dataset.models import Document


@dataclass
class Chunk:
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def create_chunk(document: Document,content: str,index: int,strategy: str,) -> Chunk:

    return Chunk(
        id=f"{document.id}:{strategy}:{index}",
        document_id=document.id,
        content=content,
        chunk_index=index,
        metadata={
            **document.metadata,
            "title": document.title,
            "source": document.source,
            "chunking_strategy": strategy,
        },
    )


def fixed_size_chunking(document: Document,chunk_size: int = 500,overlap: int = 50,) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []
    step = chunk_size - overlap

    for index, start in enumerate(range(0, len(document.content), step)):
        content = document.content[start:start + chunk_size]

        if not content:
            continue

        chunks.append(
            create_chunk(
                document=document,
                content=content,
                index=index,
                strategy="fixed_size",
            )
        )

        if start + chunk_size >= len(document.content):
            break

    return chunks


def sentence_aware_chunking(document: Document,chunk_size: int = 500,) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    sentences = re.split(
        r"(?<=[.!?])\s+",
        document.content.strip(),
    )

    chunks = []
    current_chunk = ""

    def append_chunk(content: str) -> None:
        chunks.append(
            create_chunk(
                document=document,
                content=content,
                index=len(chunks),
                strategy="sentence_aware",
            )
        )

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        
        if len(sentence) > chunk_size:
            if current_chunk:
                append_chunk(current_chunk)
                current_chunk = ""

            for start in range(0, len(sentence), chunk_size):
                append_chunk(sentence[start:start + chunk_size])

            continue

        candidate = (
            f"{current_chunk} {sentence}"
            if current_chunk
            else sentence
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            append_chunk(current_chunk)
            current_chunk = sentence

    if current_chunk:
        append_chunk(current_chunk)

    return chunks