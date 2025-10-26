# Repository Guidelines

## Project Structure & Module Organization
- `graphrag/`: core package grouped by pipeline stage (`chunker.py`, `entity_extraction.py`, `graph_builder.py`, `retrieval.py`, `pipeline.py`). Extend a stage by dropping in a new module that preserves the same function signatures.
- `scripts/`: CLI entry points. `build_graph.py` summarises the constructed graph, `query_graph.py` runs the blended vector+graph retrieval loop.
- `data/`: source corpus in `.txt`/`.md` form. Keep raw documents here; exported artefacts (e.g., GraphML) should live beside the repo root (`graphrag.graphml`).

## Build, Test, and Development Commands
- Environment setup:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Hosted agents should run project scripts via the pre-installed Miniconda interpreter:
  ```bash
  /home/rolance/miniconda3/bin/conda run -n base python <script> [args...]
  ```
  Replace `<script>` with paths like `scripts/query_graph.py` to ensure dependencies from the `base` environment are available.
- Build/inspect the knowledge graph: `python scripts/build_graph.py data [--export-graphml graphrag.graphml]`.
- Run a retrieval session: `python scripts/query_graph.py data "How does GraphRAG use knowledge graphs?"`.
- Use Gemini synthesis (defaults to `models/gemini-1.5-flash`; the loader auto-adds `models/` if missing, so variants like `gemini-flash-latest`, `gemini-1.5-flash-lite`, or `gemma-312b-it` all work):  
  `python scripts/query_graph.py data "How does GraphRAG use knowledge graphs?" --use-gemini`.
- If Gemini returns a finish-reason message instead of text, reduce `--top-k`, trim your prompt, or bump `--gemini-max-output-tokens` to avoid truncation.
- When developing modules interactively, use `python -m pip install -e .` after adding a `setup.cfg` or rely on `PYTHONPATH=.` for local imports.

## Coding Style & Naming Conventions
- Python 3.10+, four-space indentation, keep functions under ~50 lines.
- Follow PEP 8; modules and functions use `snake_case`, classes use `CamelCase`, constants use `UPPER_SNAKE_CASE`.
- Type hints are already present in the pipeline—maintain or expand them for new public APIs. Prefer concise Google-style docstrings on user-facing methods.
- Keep side effects in scripts and isolate pure logic inside `graphrag/` to ease testing.

## Testing Guidelines
- There is no automated suite yet; create `tests/` mirroring the package structure (`tests/test_chunker.py`, etc.).
- Start with `pytest` (add it to `requirements-dev.txt`) or fallback to the standard library via `python -m unittest discover tests`.
- Target at least happy-path and edge-case coverage for each pipeline stage (chunk span boundaries, empty corpora, retrieval score blends).
- Include a sample question/answer assertion in PRs by snapshotting the top-ranked chunk IDs.

## Commit & Pull Request Guidelines
- Commits should stay small and use imperative subjects, e.g., `Improve entity normalisation` or `Add graph export flag`.
- Draft PR descriptions that cover: the problem statement, a brief summary of changes, manual test commands, and before/after snippets when altering retrieval output.
- Link issues (if any) and attach console excerpts from `build_graph.py` or `query_graph.py` runs to demonstrate behaviour.
- Flag breaking changes, data schema updates, or new dependencies in a dedicated checklist so reviewers know how to reproduce your environment.

## Data & Configuration Notes
- Treat `data/` as sample content—avoid committing proprietary documents. Document provenance for any new corpus.
- When exporting graphs, keep filenames descriptive (`graphrag_<dataset>.graphml`) and add them to `.gitignore` if they are large.
- Sensitive configuration (API keys, etc.) should stay out of this repo; add values to the `.env` file (auto-loaded via `python-dotenv`) or set them as environment variables.
