"""Document loading utilities for the educational GraphRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass
class Document:
    """Simple in-memory representation of a text document."""

    doc_id: str
    title: str
    body: str


def iter_documents(path: Path) -> Iterable[Document]:
    """Yield documents from ``path`` (file or directory)."""
    if path.is_file():
        yield _load_file(path)
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield _load_file(file_path)


def load_documents(path: Path) -> List[Document]:
    """Return all documents rooted at ``path``."""
    return list(iter_documents(path))


def _load_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").title()
    return Document(doc_id=str(path), title=title, body=text)
