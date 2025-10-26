# GraphRAG Practice Lab

This repository is a self-contained playground for learning graph-based retrieval-augmented generation (GraphRAG). The code avoids external APIs and keeps the NLP components lightweight so you can focus on how knowledge graphs strengthen retrieval.

## Why GraphRAG

- **Structured reasoning**: knowledge graphs hold entities and relationships explicitly, enabling multi-hop traversal that dense vectors alone often miss.
- **Explainability**: every answer cites the graph trail that gathered supporting evidence.
- **Modularity**: each stage (chunking, extraction, graph building, retrieval, synthesis) can be swapped independently as your skills grow.

## What You Build

```
documents -> chunks -> entity extraction -> graph construction
                      |                                  |
                      v                                  v
                 embeddings                        graph retrieval
                      \                                 /
                       ----------> synthesis ----------
```

1. **Ingest documents** from `data/` (plain text or markdown).
2. **Chunk** them into overlapping passages that mimic token windows.
3. **Extract entities and relations** with a simple heuristic (capitalised n-grams plus co-occurrence).
4. **Build a heterogeneous graph** (chunks plus entities) with NetworkX.
5. **Embed chunks** using a minimal TF-IDF style bag-of-words encoder.
6. **Retrieve** by blending vector scores with short graph walks.
7. **Synthesise** a response by stitching together the highest scoring passages.

The defaults are intentionally simple so you can understand every line before replacing pieces with production-grade alternatives.

## Repository Layout

- `data/`: starter corpus describing GraphRAG concepts.
- `graphrag/`: the reusable Python package.
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
- `requirements.txt`: minimal dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Everything else uses the Python standard library.

## Quickstart

Inspect the graph:

```bash
python scripts/build_graph.py data
# optional: python scripts/build_graph.py data --export-graphml graphrag.graphml
```

Ask a question:

```bash
python scripts/query_graph.py data "How does GraphRAG use knowledge graphs?"
```

You will see an answer along with the supporting retrieval trail (chunk -> entity -> chunk).

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
