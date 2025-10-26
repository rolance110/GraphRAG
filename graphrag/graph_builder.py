"""Build a heterogeneous graph from chunks, entities, and relations."""

from __future__ import annotations

from typing import Dict, Iterable, List

import networkx as nx

from .chunker import Chunk
from .entity_extraction import Entity, Relation


Graph = nx.Graph


def build_graph(
    chunks: Iterable[Chunk],
    entities: Dict[str, Entity],
    relations: Iterable[Relation],
) -> Graph:
    graph = nx.Graph()

    for chunk in chunks:
        graph.add_node(
            chunk.chunk_id,
            type="chunk",
            doc_id=chunk.doc_id,
            text=chunk.text,
        )

    for entity in entities.values():
        graph.add_node(
            entity.entity_id,
            type="entity",
            label=entity.label,
            frequency=entity.frequency,
        )

    for chunk in chunks:
        chunk_entities = [
            entity_id
            for entity_id, entity in entities.items()
            if entity.label in chunk.text
        ]
        for entity_id in chunk_entities:
            graph.add_edge(chunk.chunk_id, entity_id, type="mentions", weight=1.0)

    for relation in relations:
        if not graph.has_node(relation.head_id) or not graph.has_node(relation.tail_id):
            continue
        graph.add_edge(
            relation.head_id,
            relation.tail_id,
            type="co_occurs",
            weight=relation.weight,
            description=relation.description,
        )
    return graph
