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
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Google Gemini to synthesise the answer text.",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Explicit Gemini API key. Defaults to the GOOGLE_GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="models/gemini-1.5-flash",
        help="Gemini model name to invoke when --use-gemini is set.",
    )
    parser.add_argument(
        "--gemini-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Gemini responses.",
    )
    parser.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=512,
        help="Maximum output tokens for Gemini responses.",
    )
    parser.add_argument(
        "--gemini-env-var",
        type=str,
        default="GOOGLE_GEMINI_API_KEY",
        help="Environment variable to read the Gemini API key from when not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = GraphRAGPipeline.from_path(args.data_path)
    if args.use_gemini:
        response = pipeline.query_with_gemini(
            args.question,
            top_k=args.top_k,
            api_key=args.gemini_api_key,
            model=args.gemini_model,
            temperature=args.gemini_temperature,
            max_output_tokens=args.gemini_max_output_tokens,
            env_var=args.gemini_env_var,
        )
    else:
        response = pipeline.query(args.question, top_k=args.top_k)
    print(response)


if __name__ == "__main__":
    main()
