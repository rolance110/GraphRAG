"""CLI helper to build the GraphRAG pipeline and inspect the graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graphrag import GraphRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphRAG artefacts and show summary.")
    parser.add_argument("data_path", type=Path, help="Path to directory or file containing source documents.")
    parser.add_argument("--export-graphml", type=Path, help="Optional path to write a GraphML snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = GraphRAGPipeline.from_path(args.data_path)
    summary = pipeline.explain_graph()
    print(summary)

    if args.export_graphml:
        graph = pipeline.artifacts.graph
        import networkx as nx_module

        nx_module.write_graphml(graph, args.export_graphml)
        print(f"Graph written to {args.export_graphml}")


if __name__ == "__main__":
    main()
