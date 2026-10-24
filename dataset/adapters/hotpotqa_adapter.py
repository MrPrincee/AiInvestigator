from dataset.models import Document
from dataset.identifiers import create_document_id


def convert_context_to_documents(example: dict) -> list[Document]:
    documents = []

    context = example["context"]

    for title, sentences in zip(
        context["title"],
        context["sentences"]
    ):
        content = " ".join(sentences).strip()

        if not content:
            continue

        document = Document(
            id=create_document_id(title, content),
            title=title,
            content=content,
            source="hotpotqa",
            metadata={
                "sentences": sentences,
                "dataset": "hotpotqa",
                "configuration": "distractor"
            }
        )

        documents.append(document)

    return documents


