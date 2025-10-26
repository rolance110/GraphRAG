"""Utilities for visualising GraphRAG graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Set

import matplotlib

# Use a non-interactive backend so rendering works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

from .graph_builder import Graph


NODE_TYPE_COLORS: Mapping[str, str] = {
    "chunk": "#4F6BED",  # blue-ish
    "entity": "#E37B40",  # orange-ish
}

EDGE_TYPE_COLORS: Mapping[str, str] = {
    "mentions": "#94A3B8",
    "co_occurs": "#F59E0B",
}


@dataclass(frozen=True)
class VisualisationConfig:
    """Options for rendering a graph snapshot."""

    layout: str = "spring"
    max_nodes: int = 200
    include_neighbors_radius: int = 2
    with_labels: bool = False
    seed: int = 42


def select_subgraph(
    graph: Graph,
    *,
    focus_nodes: Optional[Sequence[str]] = None,
    max_nodes: int = 200,
    include_neighbors_radius: int = 2,
) -> Graph:
    """Return a view of ``graph`` limited to ``max_nodes`` and optionally centred on ``focus_nodes``."""
    if graph.number_of_nodes() == 0:
        raise ValueError("The supplied graph is empty – nothing to visualise.")

    if focus_nodes:
        focus_set: Set[str] = set()
        for node in focus_nodes:
            if node not in graph:
                continue
            focus_set.add(node)
            if include_neighbors_radius <= 0:
                continue
            reachable = nx.single_source_shortest_path_length(
                graph, node, cutoff=include_neighbors_radius
            ).keys()
            focus_set.update(reachable)
        if not focus_set:
            raise ValueError("None of the focus nodes exist in the graph.")
        subgraph = graph.subgraph(focus_set)
    else:
        ordered_nodes = list(graph.nodes())
        subgraph = graph.subgraph(ordered_nodes[:max_nodes])

    if subgraph.number_of_nodes() > max_nodes:
        limited_nodes = list(subgraph.nodes())[:max_nodes]
        subgraph = subgraph.subgraph(limited_nodes)

    # Ensure we return a new Graph object rather than a graph view.
    return nx.Graph(subgraph)


def compute_layout(graph: Graph, layout: str = "spring", seed: int = 42) -> Mapping[str, tuple[float, float]]:
    """Compute node positions for ``graph`` using the requested layout."""
    layout = layout.lower()
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "spectral":
        return nx.spectral_layout(graph)
    if layout == "shell":
        return nx.shell_layout(graph)
    raise ValueError(f"Unsupported layout '{layout}'. Choose from spring, kamada_kawai, spectral, shell.")


def draw_graph(
    graph: Graph,
    output_path: Path,
    *,
    config: VisualisationConfig,
    focus_nodes: Optional[Sequence[str]] = None,
) -> Graph:
    """Render ``graph`` (or a subgraph) to ``output_path`` and return the rendered subgraph."""
    subgraph = select_subgraph(
        graph,
        focus_nodes=focus_nodes,
        max_nodes=config.max_nodes,
        include_neighbors_radius=config.include_neighbors_radius,
    )

    positions = compute_layout(subgraph, layout=config.layout, seed=config.seed)

    fig, ax = plt.subplots(figsize=_auto_size(subgraph.number_of_nodes()))
    ax.set_axis_off()

    # Draw edges grouped by type for clearer legends.
    for edge_type, color in EDGE_TYPE_COLORS.items():
        edges = [
            (u, v)
            for u, v, data in subgraph.edges(data=True)
            if data.get("type") == edge_type
        ]
        if not edges:
            continue
        nx.draw_networkx_edges(
            subgraph,
            pos=positions,
            edgelist=edges,
            edge_color=color,
            width=1.5 if edge_type == "mentions" else 1.0,
            alpha=0.6,
            ax=ax,
        )

    # Draw nodes by type.
    focus_set = set(focus_nodes or [])
    legend_handles = []
    for node_type, color in NODE_TYPE_COLORS.items():
        nodes = [
            node for node, data in subgraph.nodes(data=True) if data.get("type") == node_type
        ]
        if not nodes:
            continue
        sizes = 650 if node_type == "chunk" else 420
        collection = nx.draw_networkx_nodes(
            subgraph,
            pos=positions,
            nodelist=nodes,
            node_color=color,
            node_size=sizes,
            linewidths=1.0,
            edgecolors="#1F2933",
            alpha=0.9 if node_type == "chunk" else 0.75,
            ax=ax,
        )
        legend_handles.append((collection, node_type.title()))

    # Highlight focus nodes with a stronger border.
    if focus_set:
        highlight_nodes = [node for node in subgraph.nodes() if node in focus_set]
        if highlight_nodes:
            nx.draw_networkx_nodes(
                subgraph,
                pos=positions,
                nodelist=highlight_nodes,
                node_color="none",
                node_size=700,
                linewidths=2.6,
                edgecolors="#111827",
                ax=ax,
            )

    if config.with_labels:
        labels = _build_labels(subgraph, focus_set=focus_set)
        nx.draw_networkx_labels(
            subgraph,
            pos=positions,
            labels=labels,
            font_size=8,
            font_color="#111827",
            ax=ax,
        )

    if legend_handles:
        ax.legend(
            [handle for handle, _ in legend_handles],
            [label for _, label in legend_handles],
            loc="lower left",
            frameon=False,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return subgraph


def _auto_size(node_count: int) -> tuple[float, float]:
    """Heuristically choose a figure size based on node count."""
    if node_count < 40:
        return (8, 6)
    if node_count < 120:
        return (10, 8)
    return (12, 10)


def _build_labels(graph: Graph, *, focus_set: Set[str]) -> Mapping[str, str]:
    """Choose human-readable labels for nodes, prioritising focus nodes."""
    labels: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get("type")
        if node_type == "chunk":
            base_label = data.get("doc_id", node).split("/")[-1]
        else:
            base_label = data.get("label", node)
        if node in focus_set:
            labels[node] = base_label
        elif node_type == "entity":
            labels[node] = base_label
    return labels


__all__ = [
    "VisualisationConfig",
    "draw_graph",
    "select_subgraph",
    "compute_layout",
]
