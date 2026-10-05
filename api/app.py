"""
FastAPI REST API Module
Exposes endpoints for indexing, hybrid search, question answering, and full multi-agent extraction.
"""

import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from search.indexer import SearchIndexer
from search.hybrid_retriever import HybridRetriever, RetrievalResult
from agents.orchestrator import Orchestrator

app = FastAPI(
    title="RFP Intelligence Platform API",
    description="High-Precision RAG Search Engine and Multi-Agent RFP Processing Platform",
    version="1.0.0"
)

# Global orchestrator and retriever instances
indexer = SearchIndexer()
retriever = HybridRetriever(indexer=indexer)
orchestrator = Orchestrator(indexer=indexer, retriever=retriever)


class IndexRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute or relative path to the bid folder")
    bid_id: Optional[str] = Field(None, description="Optional custom bid identifier")


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    bid_id: Optional[str] = Field(None, description="Optional target bid identifier (e.g. Bid1, Bid2)")


class ExtractRequest(BaseModel):
    bid_folder: str = Field(..., description="Path to bid directory")
    bid_id: Optional[str] = Field(None, description="Optional bid identifier")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rfp-intelligence-platform"}


@app.get("/bids")
def list_bids():
    """Lists all currently indexed bids in the platform."""
    return {"indexed_bids": indexer.list_indexed_bids()}


@app.post("/index")
def index_bid_folder(req: IndexRequest):
    """
    Indexes a bid directory containing PDF/HTML RFP files incrementally.
    """
    if not os.path.exists(req.folder_path):
        raise HTTPException(status_code=404, detail=f"Directory '{req.folder_path}' does not exist.")

    bid_id = req.bid_id or os.path.basename(req.folder_path.rstrip("/"))
    try:
        bid_idx = orchestrator.ingestion_agent.process_bid_folder(req.folder_path, bid_id)
        return {
            "status": "success",
            "bid_id": bid_id,
            "chunks_indexed": len(bid_idx.chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_documents(
    q: str = Query(..., description="Search query string"),
    bid_id: Optional[str] = Query(None, description="Filter by bid ID"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to retrieve"),
    doc_type: Optional[str] = Query(None, description="Filter by doc_type (e.g. rfp, addendum, specs, affidavit)")
):
    """
    Executes hybrid (dense + BM25) search with reciprocal rank fusion, metadata filtering, and re-ranking.
    """
    filters = {}
    if doc_type:
        filters["doc_type"] = doc_type

    results = retriever.search(
        query=q,
        bid_id=bid_id,
        top_k=top_k,
        mode="hybrid",
        filters=filters,
        apply_rerank=True,
        apply_expansion=True
    )

    return {
        "query": q,
        "bid_id": bid_id,
        "count": len(results),
        "results": [r.model_dump() for r in results]
    }


@app.post("/ask")
def answer_question(req: AskRequest):
    """
    Answers natural language inquiries over RFP documents with strict source citations.
    """
    try:
        resp = orchestrator.answer_question(req.question, bid_id=req.bid_id)
        return resp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
def extract_bid(req: ExtractRequest):
    """
    Executes the multi-agent extraction pipeline to produce the Section 8.1 structured JSON record.
    """
    if not os.path.exists(req.bid_folder):
        raise HTTPException(status_code=404, detail=f"Directory '{req.bid_folder}' not found.")

    try:
        state = orchestrator.run_extraction_pipeline(req.bid_folder, req.bid_id)
        return state.final_output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
