import hashlib


def create_document_id(title: str, content: str) -> str:
    raw_value = f"{title}\n{content}"
    return hashlib.sha256(raw_value.encode("utf-8")).hexdigest()