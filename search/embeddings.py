"""
Dense Embeddings Module
Wraps fastembed BAAI/bge-small-en-v1.5 ONNX runtime embeddings with L2 normalization.
"""

import logging
from typing import List, Union
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger("rfp_platform.embeddings")


class EmbeddingEngine:
    """Manages dense vector generation using fastembed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"Initializing EmbeddingEngine with model {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        self.dim = 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings into normalized 2D numpy array [N, D]."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        raw_generator = self.model.embed(texts, batch_size=self.batch_size)
        embeddings = np.array(list(raw_generator), dtype=np.float32)

        # L2 normalize each row for fast dot-product cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        normalized = embeddings / norms
        return normalized

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into 1D normalized numpy array [D]."""
        emb = self.embed_texts([query])
        return emb[0]
