"""
Hybrid Retriever Module
Combines dense vector search and BM25 sparse keyword search using Reciprocal Rank Fusion (RRF),
with metadata filtering, query expansion, and cross-encoder re-ranking.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from pydantic import BaseModel, Field

from ingestion.chunker import DocumentChunk
from search.indexer import SearchIndexer, BidIndex, tokenize_for_bm25
from search.embeddings import EmbeddingEngine
from search.query_expander import QueryExpander
from search.reranker import Reranker

logger = logging.getLogger("rfp_platform.retriever")


class RetrievalResult(BaseModel):
    """Structured retrieval candidate with precise source citation."""
    chunk_id: str
    bid_id: str
    file_name: str
    page_number: int
    doc_type: str
    addendum_number: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    method: str = "hybrid_rrf"


class HybridRetriever:
    """
    Production-grade hybrid search engine:
    1. Query Expansion (procurement terminology)
    2. Dense Vector Cosine Similarity
    3. BM25 Sparse Keyword Scoring
    4. Reciprocal Rank Fusion (RRF) / Score Fusion
    5. Strict Metadata Filtering (by bid_id, doc_type, addendum_number)
    6. Cross-Encoder Re-ranking
    """

    def __init__(
        self,
        indexer: SearchIndexer,
        embedding_engine: Optional[EmbeddingEngine] = None,
        query_expander: Optional[QueryExpander] = None,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60,
        enable_reranker: bool = True
    ):
        self.indexer = indexer
        self.embedding_engine = embedding_engine or indexer.embedding_engine
        self.query_expander = query_expander or QueryExpander()
        self.reranker = reranker or (Reranker() if enable_reranker else None)
        self.rrf_k = rrf_k
        self.enable_reranker = enable_reranker

    def _matches_filters(
        self,
        chunk: DocumentChunk,
        bid_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        addendum_number: Optional[int] = None,
        is_table: Optional[bool] = None
    ) -> bool:
        """Evaluates whether a chunk satisfies the specified metadata constraints."""
        if bid_id and chunk.bid_id.lower() != bid_id.lower():
            return False
        if doc_type and chunk.doc_type.lower() != doc_type.lower():
            return False
        if addendum_number is not None and chunk.addendum_number != addendum_number:
            return False
        if is_table is not None and chunk.is_table != is_table:
            return False
        return True

    def dense_search(
        self,
        query: str,
        bid_id: Optional[str] = None,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Performs dense vector retrieval using cosine similarity."""
        filters = filters or {}
        q_vec = self.embedding_engine.embed_query(query)
        target_indices = self.indexer.get_index(bid_id)

        all_results: List[Tuple[DocumentChunk, float]] = []

        for b_idx in target_indices:
            if b_idx.embeddings is None or len(b_idx.embeddings) == 0:
                continue

            # Dot product with L2-normalized embeddings equals cosine similarity
            sims = np.dot(b_idx.embeddings, q_vec)

            for idx, score in enumerate(sims):
                chunk = b_idx.chunks[idx]
                if self._matches_filters(
                    chunk,
                    bid_id=bid_id,
                    doc_type=filters.get("doc_type"),
                    addendum_number=filters.get("addendum_number"),
                    is_table=filters.get("is_table")
                ):
                    all_results.append((chunk, float(score)))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def bm25_search(
        self,
        query: str,
        bid_id: Optional[str] = None,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Performs sparse keyword search using BM25Okapi."""
        filters = filters or {}
        q_tokens = tokenize_for_bm25(query)
        target_indices = self.indexer.get_index(bid_id)

        all_results: List[Tuple[DocumentChunk, float]] = []

        if not q_tokens:
            return []

        for b_idx in target_indices:
            if b_idx.bm25 is None or len(b_idx.chunks) == 0:
                continue

            scores = b_idx.bm25.get_scores(q_tokens)

            for idx, score in enumerate(scores):
                if score <= 0:
                    continue
                chunk = b_idx.chunks[idx]
                if self._matches_filters(
                    chunk,
                    bid_id=bid_id,
                    doc_type=filters.get("doc_type"),
                    addendum_number=filters.get("addendum_number"),
                    is_table=filters.get("is_table")
                ):
                    all_results.append((chunk, float(score)))

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def search(
        self,
        query: str,
        bid_id: Optional[str] = None,
        top_k: int = 5,
        mode: str = "hybrid",  # "hybrid", "dense", "sparse"
        filters: Optional[Dict[str, Any]] = None,
        apply_rerank: Optional[bool] = None,
        apply_expansion: bool = True
    ) -> List[RetrievalResult]:
        """
        Main retrieval entrypoint.
        Executes query expansion, dense/sparse search, RRF merge, and optional re-ranking.
        """
        filters = filters or {}
        rerank_active = apply_rerank if apply_rerank is not None else self.enable_reranker

        # 1. Query expansion
        search_query = query
        if apply_expansion and self.query_expander:
            expanded = self.query_expander.expand(query)
            # Use expanded query for search
            search_query = expanded[-1]

        # 2. Vector only mode
        if mode == "dense":
            dense_hits = self.dense_search(search_query, bid_id=bid_id, top_k=top_k * 2, filters=filters)
            if rerank_active and self.reranker:
                ranked = self.reranker.rerank(query, dense_hits, top_k=top_k)
            else:
                ranked = dense_hits[:top_k]

            return [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    bid_id=c.bid_id,
                    file_name=c.file_name,
                    page_number=c.page_number,
                    doc_type=c.doc_type,
                    addendum_number=c.addendum_number,
                    section_title=c.section_title,
                    text=c.text,
                    score=float(score),
                    dense_score=float(score),
                    method="dense"
                )
                for c, score in ranked
            ]

        # 3. Sparse only mode
        if mode == "sparse":
            sparse_hits = self.bm25_search(search_query, bid_id=bid_id, top_k=top_k * 2, filters=filters)
            if rerank_active and self.reranker:
                ranked = self.reranker.rerank(query, sparse_hits, top_k=top_k)
            else:
                ranked = sparse_hits[:top_k]

            return [
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    bid_id=c.bid_id,
                    file_name=c.file_name,
                    page_number=c.page_number,
                    doc_type=c.doc_type,
                    addendum_number=c.addendum_number,
                    section_title=c.section_title,
                    text=c.text,
                    score=float(score),
                    sparse_score=float(score),
                    method="sparse"
                )
                for c, score in ranked
            ]

        # 4. Hybrid Mode (Dense + BM25 with Reciprocal Rank Fusion)
        candidate_pool_size = max(top_k * 3, 20)
        dense_hits = self.dense_search(search_query, bid_id=bid_id, top_k=candidate_pool_size, filters=filters)
        sparse_hits = self.bm25_search(search_query, bid_id=bid_id, top_k=candidate_pool_size, filters=filters)

        # Build chunk lookup and map ranks
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_hits):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (chunk, score) in enumerate(sparse_hits):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Sort candidates by combined RRF score
        fused_candidates: List[Tuple[DocumentChunk, float]] = [
            (chunk_map[cid], score) for cid, score in rrf_scores.items()
        ]
        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        # 5. Optional Re-ranking step
        if rerank_active and self.reranker and fused_candidates:
            rerank_pool = fused_candidates[:candidate_pool_size]
            final_ranked = self.reranker.rerank(query, rerank_pool, top_k=top_k)
            method_label = "hybrid_rrf_reranked"
        else:
            final_ranked = fused_candidates[:top_k]
            method_label = "hybrid_rrf"

        results = []
        for chunk, score in final_ranked:
            cid = chunk.chunk_id
            results.append(RetrievalResult(
                chunk_id=cid,
                bid_id=chunk.bid_id,
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                doc_type=chunk.doc_type,
                addendum_number=chunk.addendum_number,
                section_title=chunk.section_title,
                text=chunk.text,
                score=float(score),
                dense_score=dense_scores_map.get(cid),
                sparse_score=sparse_scores_map.get(cid),
                method=method_label
            ))

        return results
