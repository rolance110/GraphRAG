"""CLI helper to render a GraphRAG graph snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graphrag import GraphRAGPipeline
from graphrag.visualization import VisualisationConfig, draw_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the GraphRAG knowledge graph as an image."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to directory or file containing source documents.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("graphrag_graph.png"),
        help="Path to write the rendered image (PNG). Defaults to graphrag_graph.png.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "kamada_kawai", "spectral", "shell"],
        help="Graph layout algorithm to use.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=200,
        help="Maximum number of nodes to include in the visualisation.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Number of hops to include around focus nodes.",
    )
    parser.add_argument(
        "--focus-question",
        type=str,
        default=None,
        help="Optional question used to retrieve focus nodes for highlighting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to highlight when --focus-question is provided.",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Render node labels (chunk doc IDs and entity names).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = GraphRAGPipeline.from_path(args.data_path)
    graph = pipeline.artifacts.graph

    focus_nodes = None
    if args.focus_question:
        results = pipeline.retrieve(args.focus_question, top_k=args.top_k)
        trail_nodes = []
        for result in results:
            trail_nodes.extend(result.trail)
        chunk_ids = [result.chunk_id for result in results]
        focus_nodes = list(dict.fromkeys(chunk_ids + trail_nodes))
        if not focus_nodes:
            print("No focus nodes identified from the retrieval results.")
        else:
            print(f"Highlighting {len(focus_nodes)} nodes from the retrieval trail.")

    config = VisualisationConfig(
        layout=args.layout,
        max_nodes=args.max_nodes,
        include_neighbors_radius=args.radius,
        with_labels=args.with_labels,
    )

    rendered = draw_graph(
        graph,
        args.output,
        config=config,
        focus_nodes=focus_nodes,
    )
    print(
        f"Snapshot written to {args.output} "
        f"(nodes={rendered.number_of_nodes()}, edges={rendered.number_of_edges()})."
    )


if __name__ == "__main__":
    main()
