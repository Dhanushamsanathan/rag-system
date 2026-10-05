"""
Unit Tests for RFP Chunker
"""

import pytest
from ingestion.chunker import RFPChunker, DocumentChunk
from ingestion.document_parser import ParsedPage


@pytest.fixture
def chunker():
    return RFPChunker(chunk_size=300, chunk_overlap=60, min_chunk_len=20)


def test_heading_detection(chunker):
    assert chunker.is_heading("Section 1 - General Information")
    assert chunker.is_heading("1.1 Scope of Proposal")
    assert chunker.is_heading("### Extracted Tables:")
    assert chunker.is_heading("PURPOSE OF REQUEST")
    assert not chunker.is_heading("This is a normal paragraph sentence describing hardware.")


def test_table_preservation(chunker):
    table_text = (
        "| SKU | Description | Qty |\n"
        "| --- | --- | --- |\n"
        "| 210-BLYZ | Dell Latitude 5550 | 30 |\n"
        "| WD22TB4 | Thunderbolt 4 Dock | 30 |"
    )
    page = ParsedPage(
        bid_id="TestBid",
        file_name="Specs.pdf",
        page_number=1,
        doc_type="specs",
        text=table_text,
        tables=[table_text]
    )
    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 1
    assert any(c.is_table for c in chunks)
    assert "210-BLYZ" in chunks[0].text
    assert chunks[0].bid_id == "TestBid"
