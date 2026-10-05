"""
Unit Tests for Hybrid Retrieval and Search Engine
"""

import pytest
from search.indexer import SearchIndexer, tokenize_for_bm25
from search.hybrid_retriever import HybridRetriever
from search.query_expander import QueryExpander
from ingestion.chunker import DocumentChunk


@pytest.fixture
def search_system():
    indexer = SearchIndexer()
    retriever = HybridRetriever(indexer=indexer)
    return indexer, retriever


def test_bm25_tokenization():
    tokens = tokenize_for_bm25("Dell Latitude 5550 - SKU #210-BLYZ!")
    assert "dell" in tokens
    assert "latitude" in tokens
    assert "5550" in tokens
    assert "210-blyz" in tokens


def test_query_expansion():
    expander = QueryExpander()
    expanded = expander.expand("deadline")
    assert len(expanded) >= 2
    assert "due date" in expanded[1] or "closing date" in expanded[1]


def test_retrieval_and_filtering(search_system):
    indexer, retriever = search_system
    # Index must have loaded bids
    bids = indexer.list_indexed_bids()
    assert len(bids) > 0

    # Search with Bid2 filter
    results = retriever.search("WD22TB4 dock", bid_id="Bid2", top_k=3)
    assert len(results) > 0
    assert all(r.bid_id == "Bid2" for r in results)
    assert any("wd22tb4" in r.text.lower() for r in results)
