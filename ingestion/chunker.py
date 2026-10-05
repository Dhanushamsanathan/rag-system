"""
RFP Document Chunker Module
Implements section-aware and table-aware chunking with sliding overlap,
preserving document structure, tables, and citation metadata.
"""

import re
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from ingestion.document_parser import ParsedPage

logger = logging.getLogger("rfp_platform.chunker")


class DocumentChunk(BaseModel):
    """Represents an indexable chunk with full citation traceability."""
    chunk_id: str
    bid_id: str
    file_name: str
    page_number: int
    doc_type: str
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    section_title: Optional[str] = None
    is_table: bool = False
    text: str
    token_estimate: int = Field(default=0)


class RFPChunker:
    """
    RFP-optimized chunker:
    - Preserves tables as integral semantic units without mid-row truncation
    - Detects section headings and attaches them as breadcrumbs to subsequent chunks
    - Applies sliding window character/word overlap to avoid context fragmentation
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120, min_chunk_len: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    @staticmethod
    def is_heading(line: str) -> bool:
        """Determines if a line represents an RFP heading/section boundary."""
        line = line.strip()
        if not line or len(line) > 100:
            return False
        # Markdown headings
        if line.startswith("#"):
            return True
        # RFP Section patterns
        if re.match(r"^(?:Section\s+\d+|[0-9]\.[0-9]+|\bPart\s+[A-Z]\b|SCOPE\s+AND|PURPOSE\s+OF|ARTICLE\s+\d+)", line, re.IGNORECASE):
            return True
        # ALL CAPS headings of moderate length
        if line.isupper() and len(line.split()) <= 8 and not re.match(r"^[\d\W]+$", line):
            return True
        return False

    def chunk_page(self, page: ParsedPage) -> List[DocumentChunk]:
        """Splits a single parsed page into chunks preserving section headers and tables."""
        chunks: List[DocumentChunk] = []
        raw_text = page.text
        lines = raw_text.split("\n")

        current_section = None
        current_block: List[str] = []
        in_table = False
        table_block: List[str] = []

        chunk_counter = 0

        def emit_chunk(content: str, is_tbl: bool = False, sec_title: Optional[str] = None):
            nonlocal chunk_counter
            content = content.strip()
            if len(content) < self.min_chunk_len:
                return

            # Prefix chunk with section breadcrumb if available and not already in text
            final_text = content
            if sec_title and not content.startswith(sec_title):
                final_text = f"[{sec_title}]\n{content}"

            chunk_id = f"{page.bid_id}_{page.file_name}_p{page.page_number}_c{chunk_counter}"
            chunk_counter += 1

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                bid_id=page.bid_id,
                file_name=page.file_name,
                page_number=page.page_number,
                doc_type=page.doc_type,
                addendum_number=page.addendum_number,
                document_date=page.document_date,
                section_title=sec_title,
                is_table=is_tbl,
                text=final_text,
                token_estimate=len(final_text.split())
            ))

        for line in lines:
            line_str = line.strip()

            # Handle Markdown table lines
            if line_str.startswith("|") and line_str.endswith("|"):
                in_table = True
                table_block.append(line_str)
                continue
            elif in_table:
                # End of table block
                in_table = False
                tbl_text = "\n".join(table_block)
                if len(tbl_text) > self.chunk_size * 2:
                    # Table is very large, split into manageable sub-blocks preserving header
                    header = table_block[:2]
                    rows = table_block[2:]
                    sub_rows = []
                    curr_len = 0
                    for r in rows:
                        sub_rows.append(r)
                        curr_len += len(r)
                        if curr_len >= self.chunk_size:
                            sub_tbl = "\n".join(header + sub_rows)
                            emit_chunk(sub_tbl, is_tbl=True, sec_title=current_section)
                            sub_rows = []
                            curr_len = 0
                    if sub_rows:
                        emit_chunk("\n".join(header + sub_rows), is_tbl=True, sec_title=current_section)
                else:
                    emit_chunk(tbl_text, is_tbl=True, sec_title=current_section)
                table_block = []

            # Check for section header
            if self.is_heading(line_str):
                # Flush previous block if any
                if current_block:
                    self._sliding_chunk("\n".join(current_block), current_section, emit_chunk)
                    current_block = []
                current_section = line_str
                continue

            current_block.append(line_str)

        # Flush any trailing table
        if in_table and table_block:
            emit_chunk("\n".join(table_block), is_tbl=True, sec_title=current_section)

        # Flush any trailing text block
        if current_block:
            self._sliding_chunk("\n".join(current_block), current_section, emit_chunk)

        return chunks

    def _sliding_chunk(self, text: str, sec_title: Optional[str], emit_func):
        """Applies sliding window chunking with overlap across textual paragraphs."""
        paragraphs = text.split("\n\n")
        curr_chunk: List[str] = []
        curr_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If single paragraph exceeds chunk size, split by sentences
            if len(para) > self.chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if curr_len + len(sent) > self.chunk_size and curr_chunk:
                        chunk_text = " ".join(curr_chunk)
                        emit_func(chunk_text, is_tbl=False, sec_title=sec_title)
                        # Keep overlap from end of chunk
                        overlap_words = " ".join(curr_chunk).split()[-15:]
                        curr_chunk = [" ".join(overlap_words)] if overlap_words else []
                        curr_len = sum(len(w) for w in curr_chunk)
                    curr_chunk.append(sent)
                    curr_len += len(sent)
            else:
                if curr_len + len(para) > self.chunk_size and curr_chunk:
                    chunk_text = "\n".join(curr_chunk)
                    emit_func(chunk_text, is_tbl=False, sec_title=sec_title)
                    # Keep overlap
                    overlap_words = "\n".join(curr_chunk).split()[-15:]
                    curr_chunk = [" ".join(overlap_words)] if overlap_words else []
                    curr_len = sum(len(w) for w in curr_chunk)
                curr_chunk.append(para)
                curr_len += len(para)

        if curr_chunk:
            emit_func("\n".join(curr_chunk), is_tbl=False, sec_title=sec_title)

    def chunk_documents(self, pages: List[ParsedPage]) -> List[DocumentChunk]:
        """Processes all parsed pages and returns indexable chunks."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        logger.info(f"Generated {len(all_chunks)} chunks from {len(pages)} pages.")
        return all_chunks
