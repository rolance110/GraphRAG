"""Lightweight embedding and similarity utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .chunker import Chunk


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class Embedding:
    key: str
    vector: List[float]


class BagOfWordsEmbedder:
    """A TF-IDF-like embedder implemented without external dependencies."""

    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []

    def fit(self, documents: Iterable[str]) -> None:
        doc_tokens = [set(tokenize(doc)) for doc in documents]
        vocab = sorted({token for tokens in doc_tokens for token in tokens})
        self.vocabulary = {token: idx for idx, token in enumerate(vocab)}

        doc_count = len(doc_tokens)
        self.idf = []
        for token in vocab:
            containing = sum(1 for tokens in doc_tokens if token in tokens)
            # add-one smoothing
            score = math.log(1 + doc_count / (1 + containing))
            self.idf.append(score)

    def transform(self, text: str) -> List[float]:
        if not self.vocabulary:
            raise RuntimeError("Embedder must be fit before calling transform().")
        token_counts = Counter(tokenize(text))
        vector = [0.0] * len(self.vocabulary)
        max_count = max(token_counts.values()) if token_counts else 1
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
            idx = self.vocabulary[token]
            tf = 0.5 + 0.5 * (count / max_count)
            vector[idx] = tf * self.idf[idx]
        return vector

    def fit_transform_chunks(self, chunks: Iterable[Chunk]) -> Dict[str, Embedding]:
        chunk_list = list(chunks)
        self.fit(chunk.text for chunk in chunk_list)
        embeddings = {
            chunk.chunk_id: Embedding(key=chunk.chunk_id, vector=self.transform(chunk.text))
            for chunk in chunk_list
        }
        return embeddings


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
