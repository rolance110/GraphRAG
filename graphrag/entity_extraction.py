"""Naive entity and relation extraction for educational purposes."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence

from .chunker import Chunk


ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


@dataclass
class Entity:
    entity_id: str
    label: str
    frequency: int


@dataclass
class Relation:
    head_id: str
    tail_id: str
    weight: float
    description: str


def extract_entities(chunks: Iterable[Chunk], min_freq: int = 2) -> Dict[str, Entity]:
    """Extract entities by counting capitalized n-grams."""
    counter: Counter[str] = Counter()
    for chunk in chunks:
        matches = ENTITY_PATTERN.findall(chunk.text)
        counter.update(matches)

    entities: Dict[str, Entity] = {}
    for label, freq in counter.items():
        if freq < min_freq:
            continue
        entity_id = label.lower().replace(" ", "_")
        entities[entity_id] = Entity(entity_id=entity_id, label=label, frequency=freq)
    return entities


def extract_relations(
    chunks: Iterable[Chunk],
    entities: Dict[str, Entity],
) -> List[Relation]:
    """Create co-occurrence relations between entities within chunks."""
    relations: Dict[tuple[str, str], int] = defaultdict(int)

    entity_labels = {entity.label: entity_id for entity_id, entity in entities.items()}
    for chunk in chunks:
        matches = [
            entity_labels[m]
            for m in ENTITY_PATTERN.findall(chunk.text)
            if m in entity_labels
        ]
        for head, tail in combinations(sorted(set(matches)), 2):
            relations[(head, tail)] += 1

    relation_objs: List[Relation] = []
    for (head, tail), weight in relations.items():
        description = f"Co-occurs with {entities[tail].label}"
        relation_objs.append(Relation(head_id=head, tail_id=tail, weight=float(weight), description=description))

    return relation_objs
