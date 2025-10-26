# GraphRAG Practice Lab

GraphRAG Practice Lab is a self-contained playground for understanding how knowledge graphs strengthen retrieval-augmented generation (RAG). Every component ships with transparent, lightweight implementations so you can inspect the full pipeline and progressively swap in production-ready alternatives.

## Project Overview

**Goals**
- demystify graph-aware retrieval by walking through a complete, reproducible pipeline
- highlight where graphs add explainability and multi-hop reasoning beyond dense vectors
- provide modular hooks so you can iterate on individual stages without rewriting the system

**Key Features**
- Structured reasoning via explicit entities and relationships
- End-to-end transparency: every answer surfaces the supporting trail
- Drop-in extensibility across chunking, extraction, embedding, graph building, retrieval, and synthesis

### Pipeline at a Glance

```
documents -> chunks -> entity extraction -> graph construction
                      |                                  |
                      v                                  v
                 embeddings                        graph retrieval
                      \                                 /
                       ----------> synthesis ----------
```

1. **Load documents** from `data/` (plain text or markdown) with `data_loader.py`.
2. **Chunk** into overlapping windows in `chunker.py` to mimic model token limits.
3. **Extract entities and relations** using heuristic capitalised n-grams in `entity_extraction.py`.
4. **Build the heterogeneous graph** of chunks and entities via NetworkX in `graph_builder.py`.
5. **Embed chunks** with a handcrafted TF-IDF encoder in `embedder.py`.
6. **Blend retrieval scores** by combining vector similarity and short graph walks in `retrieval.py`.
7. **Synthesise responses** by stitching together the highest scoring context in `pipeline.py`.

The defaults are intentionally simple so you can understand every line before replacing pieces with production-grade alternatives.

## Repository Layout

- `data/`: starter corpus describing GraphRAG concepts.
- `graphrag/`: reusable Python package organised by pipeline stage.
  - `data_loader.py`: loads `.txt`/`.md` files into memory.
  - `chunker.py`: turns documents into overlapping windows.
  - `entity_extraction.py`: naive entity and relation finder.
  - `embedder.py`: handcrafted TF-IDF style embeddings and cosine similarity.
  - `graph_builder.py`: constructs the NetworkX graph.
  - `retrieval.py`: graph-aware retrieval with trail tracking.
  - `pipeline.py`: orchestrates the full GraphRAG pipeline.
- `scripts/`: command-line helpers.
  - `build_graph.py`: inspect the graph summary (and optionally export GraphML).
  - `query_graph.py`: run an interactive style question.
- `requirements.txt`: minimal dependencies for the playground.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Everything else relies on the Python standard library. For editable installs during development, run `python -m pip install -e .` or set `PYTHONPATH=.` when invoking scripts.

## Command Reference

- **Build & Inspect the Graph**
  ```bash
  python scripts/build_graph.py data
  # Optional export for Gephi or Neo4j Bloom:
  python scripts/build_graph.py data --export-graphml graphrag.graphml
  ```

- **Query the GraphRAG Pipeline**
  ```bash
  python scripts/query_graph.py data "How does GraphRAG use knowledge graphs?"
  ```
  Add `--use-gemini` to synthesise with Gemini models (defaults to `models/gemini-1.5-flash`). If the model returns a finish-reason notice, reduce `--top-k` or increase `--gemini-max-output-tokens`.

- **Visualise the Graph**
  ```bash
  python scripts/visualize_graph.py data --output graphrag_graph.png --layout spring
  ```
  Supply `--focus-question "GraphRAG basics?" --top-k 5` to highlight the retrieval trail for a specific query, or `--with-labels` to print chunk/entity labels on the graph.

## Testing

Tests live in `tests/` and mirror the package structure (e.g. `tests/test_chunker.py`). Start by installing a test runner:

```bash
python -m pip install pytest  # or add pytest to requirements-dev.txt
```

Run the full suite with Pytest:

```bash
pytest
```

If you prefer the standard library tools or do not have Pytest available, fall back to:

```bash
python -m unittest discover tests
```

Aim to cover happy-path flows plus edge cases such as empty corpora, chunk boundary handling, and retrieval score blending. Snapshot top-ranked chunk IDs when you add regression questions to guard against accidental behaviour changes.

## Practice Ideas

1. **Upgrade extraction**: plug in spaCy, OpenAI function calling, or a ruleset tailored to your documents.
2. **Improve retrieval**: experiment with weighted graph traversals, personalised PageRank, or mixing dense vector databases.
3. **Strengthen synthesis**: call your preferred LLM and supply the retrieved context.
4. **Evaluate**: design a small Q&A set and compare pure vector search versus graph-augmented answers.

## Learning Checklist
- [ ] Understand how GraphRAG differs from vanilla RAG.
- [ ] Be able to explain each pipeline stage with code references.
- [ ] Run the scripts and inspect graph nodes and edges.
- [ ] Swap one component (chunker, extractor, retrieval) and observe the change.
- [ ] Export the graph and visualise it in Gephi or Neo4j Bloom.

## Next Steps

When you feel comfortable, extend this lab into a real project: ingest your own data, replace heuristics with robust NLP models, and connect the pipeline to a hosted LLM for natural language synthesis. The modular design makes those upgrades straightforward. Happy experimenting!
