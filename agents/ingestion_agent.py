"""
Ingestion Agent Module
Scans bid directories, detects file types, extracts tables, and triggers indexing.
"""

import time
import logging
from typing import List, Optional

from ingestion.document_parser import DocumentParser, ParsedPage
from ingestion.chunker import RFPChunker, DocumentChunk
from search.indexer import SearchIndexer, BidIndex
from agents.state import SharedAgentState

logger = logging.getLogger("rfp_platform.ingestion_agent")


class IngestionAgent:
    """Agent orchestrating document discovery, parsing, chunking, and index generation."""

    def __init__(self, parser: Optional[DocumentParser] = None, chunker: Optional[RFPChunker] = None, indexer: Optional[SearchIndexer] = None):
        self.parser = parser or DocumentParser()
        self.chunker = chunker or RFPChunker()
        self.indexer = indexer or SearchIndexer()

    def process_bid_folder(self, folder_path: str, bid_id: str, state: Optional[SharedAgentState] = None) -> BidIndex:
        """Parses files in bid directory, generates chunks, and updates the search index."""
        start_t = time.time()
        logger.info(f"IngestionAgent starting processing for {bid_id} at {folder_path}...")

        # 1. Parse documents
        parsed_pages: List[ParsedPage] = self.parser.parse_bid_directory(folder_path, bid_id)

        # 2. Chunk documents
        chunks: List[DocumentChunk] = self.chunker.chunk_documents(parsed_pages)

        # 3. Incremental indexing
        bid_index = self.indexer.index_bid(bid_id, chunks)

        latency = (time.time() - start_t) * 1000
        logger.info(f"IngestionAgent completed {bid_id}: {len(parsed_pages)} pages, {len(chunks)} chunks in {latency:.2f}ms")

        if state:
            state.add_trace(
                agent_name="IngestionAgent",
                action="process_bid_folder",
                input_data={"folder_path": folder_path, "bid_id": bid_id},
                output_data={"pages_count": len(parsed_pages), "chunks_count": len(chunks)},
                latency_ms=latency
            )

        return bid_index
