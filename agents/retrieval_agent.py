"""
Retrieval Agent Module
Acts as a specialized tool for other agents: performs query expansion, metadata filtering,
dense+BM25 hybrid retrieval, and citation structuring.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from search.hybrid_retriever import HybridRetriever, RetrievalResult
from agents.state import Citation, SharedAgentState

logger = logging.getLogger("rfp_platform.retrieval_agent")


class RetrievalAgent:
    """Specialized search agent serving evidence with traceable citations to other agents."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def retrieve_evidence(
        self,
        query: str,
        bid_id: str,
        top_k: int = 5,
        doc_type: Optional[str] = None,
        addendum_number: Optional[int] = None,
        is_table: Optional[bool] = None,
        state: Optional[SharedAgentState] = None
    ) -> List[RetrievalResult]:
        """
        Executes query on search index with optional metadata restrictions,
        returning top ranked results and logging to state observability trace.
        """
        start_t = time.time()
        filters = {}
        if doc_type:
            filters["doc_type"] = doc_type
        if addendum_number is not None:
            filters["addendum_number"] = addendum_number
        if is_table is not None:
            filters["is_table"] = is_table

        results = self.retriever.search(
            query=query,
            bid_id=bid_id,
            top_k=top_k,
            mode="hybrid",
            filters=filters,
            apply_rerank=True,
            apply_expansion=True
        )

        latency = (time.time() - start_t) * 1000

        if state:
            state.add_trace(
                agent_name="RetrievalAgent",
                action="retrieve_evidence",
                input_data={"query": query, "bid_id": bid_id, "filters": filters, "top_k": top_k},
                output_data={"results_count": len(results), "top_score": results[0].score if results else 0.0},
                latency_ms=latency
            )

        return results

    @staticmethod
    def to_citations(results: List[RetrievalResult], max_citations: int = 2) -> List[Citation]:
        """Converts retrieval results into citation models."""
        citations = []
        for r in results[:max_citations]:
            snippet = r.text.strip().replace("\n", " ")
            if len(snippet) > 250:
                snippet = snippet[:247] + "..."
            citations.append(Citation(
                file=r.file_name,
                page=r.page_number,
                text_snippet=snippet,
                score=round(r.score, 4)
            ))
        return citations
