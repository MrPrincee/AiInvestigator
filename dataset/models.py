from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)