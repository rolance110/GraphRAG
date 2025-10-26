"""Educational GraphRAG toolkit.

This package implements a minimal end-to-end graph-based
retrieval-augmented generation pipeline suitable for learning.
"""

from .pipeline import GraphRAGPipeline
from .visualization import VisualisationConfig, draw_graph

__all__ = ["GraphRAGPipeline", "VisualisationConfig", "draw_graph"]
