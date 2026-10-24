from collections.abc import Iterable, Iterator

from dataset.models import Document


def deduplicate_documents(documents: Iterable[Document]) -> Iterator[Document]:
    seen_ids = set()

    for document in documents:
        if document.id in seen_ids:
            continue

        seen_ids.add(document.id)
        yield document