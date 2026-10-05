"""
Semantic Cache & Latency/Cost Tracking Module
Caches repeated queries using cosine similarity of query embeddings,
recording latency improvements and token savings.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

logger = logging.getLogger("rfp_platform.semantic_cache")


class SemanticCache:
    """
    Semantic cache storing query embeddings and prior responses.
    Matches queries using cosine similarity with a configurable similarity threshold.
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold
        self.cache_entries: List[Dict[str, Any]] = []
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saved_latency_ms": 0.0,
            "saved_tokens": 0
        }

    def lookup(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[Any, float]]:
        """
        Searches cache for semantically similar previous queries.
        Returns (cached_result, similarity_score) if match found.
        """
        if not self.cache_entries:
            self.stats["misses"] += 1
            return None

        best_sim = -1.0
        best_entry = None

        for entry in self.cache_entries:
            sim = float(np.dot(entry["embedding"], query_embedding))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.similarity_threshold and best_entry is not None:
            self.stats["hits"] += 1
            saved_time = best_entry.get("original_latency_ms", 150.0)
            self.stats["saved_latency_ms"] += saved_time
            self.stats["saved_tokens"] += best_entry.get("token_count", 250)
            logger.info(f"Semantic Cache HIT (sim={best_sim:.4f}) for: '{query[:40]}...'")
            return best_entry["response"], best_sim

        self.stats["misses"] += 1
        return None

    def store(self, query: str, query_embedding: np.ndarray, response: Any, original_latency_ms: float = 0.0, token_count: int = 250):
        """Stores a query, its normalized embedding, and response into cache."""
        self.cache_entries.append({
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "original_latency_ms": original_latency_ms,
            "token_count": token_count,
            "timestamp": time.time()
        })
        logger.info(f"Stored query in semantic cache (Total entries: {len(self.cache_entries)})")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns cache efficiency metrics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total) if total > 0 else 0.0
        return {
            "total_queries": total,
            "cache_hits": self.stats["hits"],
            "cache_misses": self.stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "cumulative_latency_saved_ms": round(self.stats["saved_latency_ms"], 2),
            "estimated_tokens_saved": self.stats["saved_tokens"]
        }
