"""
Dual Vector & Sparse Keyword Indexer
Supports dense embedding matrix, BM25Okapi keyword index, and incremental bid-level persistence.
"""

import os
import re
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.chunker import DocumentChunk
from search.embeddings import EmbeddingEngine

logger = logging.getLogger("rfp_platform.indexer")


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens for BM25 keyword matching."""
    text_clean = re.sub(r"[^\w\s-]", " ", text.lower())
    return [t for t in text_clean.split() if len(t) > 1]


class BidIndex:
    """Holds dense and sparse search indices for a single bid or collection of bids."""

    def __init__(self, bid_id: str):
        self.bid_id = bid_id
        self.chunks: List[DocumentChunk] = []
        self.chunk_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[DocumentChunk], embedding_engine: EmbeddingEngine):
        """Builds both dense and BM25 representations for the chunks."""
        self.chunks = chunks
        self.chunk_ids = [c.chunk_id for c in chunks]

        # 1. Build dense embeddings
        texts = [c.text for c in chunks]
        logger.info(f"Generating dense embeddings for {len(chunks)} chunks in {self.bid_id}...")
        self.embeddings = embedding_engine.embed_texts(texts)

        # 2. Build BM25 keyword index
        logger.info(f"Building BM25 sparse index for {len(chunks)} chunks in {self.bid_id}...")
        self.tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, directory: str):
        """Persists the bid index to disk."""
        os.makedirs(directory, exist_ok=True)
        out_path = os.path.join(directory, f"{self.bid_id}_index.pkl")
        data = {
            "bid_id": self.bid_id,
            "chunks": [c.model_dump() for c in self.chunks],
            "chunk_ids": self.chunk_ids,
            "embeddings": self.embeddings,
            "tokenized_corpus": self.tokenized_corpus
        }
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved index for {self.bid_id} to {out_path}")

    @classmethod
    def load(cls, file_path: str) -> "BidIndex":
        """Loads index from disk."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        idx = cls(bid_id=data["bid_id"])
        idx.chunks = [DocumentChunk(**c) for c in data["chunks"]]
        idx.chunk_ids = data["chunk_ids"]
        idx.embeddings = data["embeddings"]
        idx.tokenized_corpus = data["tokenized_corpus"]
        idx.bm25 = BM25Okapi(idx.tokenized_corpus)
        logger.info(f"Loaded index for {idx.bid_id} ({len(idx.chunks)} chunks) from {file_path}")
        return idx


class SearchIndexer:
    """
    Coordinator for incremental multi-bid search indices.
    Allows indexing each bid independently without re-indexing existing bids.
    """

    def __init__(self, storage_dir: str = "./index_storage", embedding_engine: Optional[EmbeddingEngine] = None):
        self.storage_dir = storage_dir
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.indices: Dict[str, BidIndex] = {}
        os.makedirs(self.storage_dir, exist_ok=True)
        self.load_all_existing()

    def load_all_existing(self):
        """Scans storage directory and loads all pre-computed bid indices."""
        if not os.path.exists(self.storage_dir):
            return
        for fname in os.listdir(self.storage_dir):
            if fname.endswith("_index.pkl"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    idx = BidIndex.load(fpath)
                    self.indices[idx.bid_id] = idx
                except Exception as e:
                    logger.error(f"Error loading index {fname}: {e}")

    def index_bid(self, bid_id: str, chunks: List[DocumentChunk], force_reindex: bool = False) -> BidIndex:
        """
        Indexes a single bid incrementally.
        If already indexed and not force_reindex, loads from cache.
        """
        cache_path = os.path.join(self.storage_dir, f"{bid_id}_index.pkl")
        if not force_reindex and os.path.exists(cache_path) and bid_id in self.indices:
            logger.info(f"Bid {bid_id} already indexed. Using cached index.")
            return self.indices[bid_id]

        idx = BidIndex(bid_id=bid_id)
        idx.build(chunks, self.embedding_engine)
        idx.save(self.storage_dir)
        self.indices[bid_id] = idx
        return idx

    def get_index(self, bid_id: Optional[str] = None) -> List[BidIndex]:
        """Returns specific bid index or all available bid indices."""
        if bid_id:
            if bid_id in self.indices:
                return [self.indices[bid_id]]
            cache_path = os.path.join(self.storage_dir, f"{bid_id}_index.pkl")
            if os.path.exists(cache_path):
                idx = BidIndex.load(cache_path)
                self.indices[bid_id] = idx
                return [idx]
            return []
        return list(self.indices.values())

    def list_indexed_bids(self) -> List[str]:
        return list(self.indices.keys())
