"""
Re-ranking Module
Applies cross-encoder re-ranking to top candidate chunks for high-precision ordering.
"""

import logging
from typing import List, Tuple
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ingestion.chunker import DocumentChunk

logger = logging.getLogger("rfp_platform.reranker")


class Reranker:
    """Cross-encoder re-ranker for RFP passages."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            logger.info(f"Loading TextCrossEncoder ({self.model_name})...")
            self.model = TextCrossEncoder(model_name=self.model_name)
            logger.info("TextCrossEncoder loaded.")
        except Exception as e:
            logger.warning(f"Could not load ONNX cross-encoder ({e}). Will use heuristic rank-blending.")
            self.model = None

    def rerank(self, query: str, candidates: List[Tuple[DocumentChunk, float]], top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Takes candidate (chunk, initial_score) tuples, computes cross-encoder scores,
        and returns the top_k re-ordered candidates.
        """
        if not candidates:
            return []

        chunks = [c[0] for c in candidates]
        passages = [c.text for c in chunks]

        if self.model is not None:
            try:
                scores = list(self.model.rerank(query, passages))
                ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception as e:
                logger.warning(f"Error during cross-encoder execution ({e}), falling back to initial scores.")

        # Fallback to initial scores
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)
        return candidates_sorted[:top_k]
