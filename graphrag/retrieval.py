"""Graph-aware retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx

from .chunker import Chunk
from .embedder import BagOfWordsEmbedder, cosine_similarity


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    text: str
    trail: List[str]


class GraphRetriever:
    def __init__(
        self,
        graph: nx.Graph,
        chunk_index: Dict[str, Chunk],
        embedder: BagOfWordsEmbedder,
        chunk_embeddings: Dict[str, List[float]],
    ):
        self.graph = graph
        self.chunk_index = chunk_index
        self.embedder = embedder
        self.chunk_embeddings = chunk_embeddings

    def query(self, text: str, k: int = 5) -> List[RetrievalResult]:
        query_vec = self.embedder.transform(text)
        scored_chunks = self._score_chunks(query_vec)
        top_chunks = scored_chunks[:k]
        expanded = self._expand_via_graph(top_chunks)
        return expanded

    def _score_chunks(self, query_vec: List[float]) -> List[Tuple[str, float]]:
        scores = []
        for chunk_id, embedding in self.chunk_embeddings.items():
            score = cosine_similarity(query_vec, embedding)
            scores.append((chunk_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def _expand_via_graph(self, top_chunks: List[Tuple[str, float]]) -> List[RetrievalResult]:
        results: List[RetrievalResult] = []
        visited = set()

        for chunk_id, base_score in top_chunks:
            if chunk_id not in self.graph:
                continue
            trail = [chunk_id]
            score = base_score
            for neighbor in self.graph.neighbors(chunk_id):
                edge = self.graph[chunk_id][neighbor]
                if edge.get("type") != "mentions":
                    continue
                for expansion in self.graph.neighbors(neighbor):
                    if expansion == chunk_id:
                        continue
                    if expansion in visited:
                        continue
                    if self.graph.nodes[expansion].get("type") != "chunk":
                        continue
                    expanded_score = base_score * 0.8
                    results.append(
                        RetrievalResult(
                            chunk_id=expansion,
                            score=expanded_score,
                            text=self.chunk_index[expansion].text,
                            trail=[chunk_id, neighbor, expansion],
                        )
                    )
                    visited.add(expansion)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=self.chunk_index[chunk_id].text,
                    trail=trail,
                )
            )
            visited.add(chunk_id)

        results.sort(key=lambda r: r.score, reverse=True)
        return results
