"""High level GraphRAG pipeline orchestrating ingestion and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .chunker import Chunk, chunk_corpus
from .data_loader import Document, load_documents
from .embedder import BagOfWordsEmbedder
from .entity_extraction import Entity, extract_entities, extract_relations
from .graph_builder import Graph, build_graph
from .gemini import GeminiAnswerGenerator, GeminiConfig
from .retrieval import GraphRetriever, RetrievalResult


@dataclass
class PipelineArtifacts:
    documents: List[Document]
    chunks: List[Chunk]
    entities: Dict[str, Entity]
    graph: Graph
    chunk_embeddings: Dict[str, List[float]]
    embedder: BagOfWordsEmbedder


class GraphRAGPipeline:
    """User-facing API for practicing GraphRAG."""

    def __init__(self, artifacts: PipelineArtifacts):
        self.artifacts = artifacts

        chunk_index = {chunk.chunk_id: chunk for chunk in artifacts.chunks}
        self.retriever = GraphRetriever(
            graph=artifacts.graph,
            chunk_index=chunk_index,
            embedder=artifacts.embedder,
            chunk_embeddings=artifacts.chunk_embeddings,
        )

    @classmethod
    def from_path(cls, path: Path) -> "GraphRAGPipeline":
        documents = load_documents(path)
        chunks = chunk_corpus(documents)

        entities = extract_entities(chunks)
        relations = extract_relations(chunks, entities)

        graph = build_graph(chunks, entities, relations)

        embedder = BagOfWordsEmbedder()
        embedder.fit(chunk.text for chunk in chunks)
        chunk_embeddings = {
            chunk.chunk_id: embedder.transform(chunk.text)
            for chunk in chunks
        }

        artifacts = PipelineArtifacts(
            documents=documents,
            chunks=chunks,
            entities=entities,
            graph=graph,
            chunk_embeddings=chunk_embeddings,
            embedder=embedder,
        )
        return cls(artifacts)

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        return self.retriever.query(question, k=top_k)

    def query(self, question: str, top_k: int = 3) -> str:
        results = self.retrieve(question, top_k=top_k)
        answer = self._synthesise_answer(question, results)
        context = "\n\n".join(
            f"[{idx+1}] Score={result.score:.2f} Trail -> {' -> '.join(result.trail)}\n{result.text}"
            for idx, result in enumerate(results[:top_k])
        )
        return f"Question: {question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"

    def query_with_gemini(
        self,
        question: str,
        top_k: int = 3,
        *,
        api_key: Optional[str] = None,
        model: str = "models/gemini-1.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        env_var: str = "GOOGLE_GEMINI_API_KEY",
    ) -> str:
        results = self.retrieve(question, top_k=top_k)
        contexts = [result.text for result in results[:top_k]]
        if api_key:
            config = GeminiConfig(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        else:
            config = GeminiConfig.from_env(
                env_var=env_var,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        generator = GeminiAnswerGenerator(config)
        answer = generator.answer(question, contexts)
        context = "\n\n".join(
            f"[{idx+1}] Score={result.score:.2f} Trail -> {' -> '.join(result.trail)}\n{result.text}"
            for idx, result in enumerate(results[:top_k])
        )
        return f"Question: {question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"

    def explain_graph(self) -> str:
        node_counts = {}
        for _, data in self.artifacts.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        lines = [
            "Graph Summary",
            "------------",
        ]
        for node_type, count in sorted(node_counts.items()):
            lines.append(f"{node_type.title()} nodes: {count}")
        lines.append(f"Edges: {self.artifacts.graph.number_of_edges()}")
        lines.append(f"Documents: {len(self.artifacts.documents)}")
        lines.append(f"Chunks: {len(self.artifacts.chunks)}")
        return "\n".join(lines)

    def _synthesise_answer(self, question: str, results: Sequence[RetrievalResult]) -> str:
        if not results:
            return "No relevant information found."
        bullet_points = []
        for result in results[:3]:
            snippet = result.text.strip().split(". ")[0]
            bullet_points.append(f"- {snippet}")
        return "\n".join(bullet_points)
