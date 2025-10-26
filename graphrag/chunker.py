"""Document chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .data_loader import Document


@dataclass
class Chunk:
    """Atomic unit that the graph ingests."""

    chunk_id: str
    doc_id: str
    text: str


def chunk_document(
    document: Document,
    max_tokens: int = 120,
    overlap: int = 20,
) -> List[Chunk]:
    """Split a document into overlapping character-based chunks.

    A conservative character limit approximates tokens without
    depending on an external tokenizer.
    """

    tokens = document.body.split()
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk_tokens = tokens[start:end]
        text = " ".join(chunk_tokens)
        chunk_id = f"{document.doc_id}::chunk-{idx}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=document.doc_id, text=text))
        if end == len(tokens):
            break
        start = end - overlap
        idx += 1
    return chunks


def chunk_corpus(documents: Iterable[Document]) -> List[Chunk]:
    """Chunk every document in ``documents``."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
