def build_sources(results: list[dict]) -> list[dict]:
    sources = []
    seen_chunks = set()

    for result in results:
        chunk_id = result.get("chunk_id")
        document_id = result.get("document_id")
        content = result.get("content")

        if not chunk_id or not document_id or not content:
            continue

        if chunk_id in seen_chunks:
            continue

        seen_chunks.add(chunk_id)

        sources.append(
            {
                "citation_id": len(sources) + 1,
                "document_id": document_id,
                "document_title": result.get("title"),
                "source": result.get("source"),
                "chunk_id": chunk_id,
                "chunk_index": result.get("chunk_index"),
                "page_number": result.get("page_number"),
                "evidence": content,
                "retrieval_score": result.get("score"),
            }
        )

    return sources


def build_context(sources: list[dict]) -> str:
    blocks = []

    for source in sources:
        blocks.append(
            f"[{source['citation_id']}]\n"
            f"Document: {source['document_title']}\n"
            f"Document ID: {source['document_id']}\n"
            f"Page: {source['page_number']}\n"
            f"Evidence:\n{source['evidence']}"
        )

    return "\n\n".join(blocks)