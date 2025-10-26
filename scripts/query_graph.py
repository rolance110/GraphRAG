"""CLI helper to query the GraphRAG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graphrag import GraphRAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a GraphRAG pipeline.")
    parser.add_argument("data_path", type=Path, help="Path to source documents.")
    parser.add_argument("question", type=str, help="Natural language question to ask.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of contexts to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = GraphRAGPipeline.from_path(args.data_path)
    response = pipeline.query(args.question, top_k=args.top_k)
    print(response)


if __name__ == "__main__":
    main()
