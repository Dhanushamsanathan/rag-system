"""
Document Ingestion & Parsing Module
Extracts clean text, structured tables, and rich metadata from PDF and HTML documents.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger("rfp_platform.ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ParsedPage(BaseModel):
    """Represents an extracted page or logical document section."""
    bid_id: str
    file_name: str
    page_number: int  # 1-indexed
    doc_type: str     # bid_page, rfp, addendum, specs, affidavit
    addendum_number: Optional[int] = None
    document_date: Optional[str] = None
    text: str
    tables: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentParser:
    """Robust parser for RFP PDF and HTML files."""

    def __init__(self):
        pass

    @staticmethod
    def detect_doc_type(file_name: str, content_preview: str = "") -> Tuple[str, Optional[int]]:
        """
        Classifies document type and extracts addendum number.
        Types: bid_page, rfp, addendum, specs, affidavit
        """
        fn = file_name.lower()
        content = content_preview.lower()

        # Check for addendum first
        addendum_match = re.search(r"addendum\s*(?:no\.?|#)?\s*(\d+)", fn) or re.search(r"addendum\s*(?:no\.?|#)?\s*(\d+)", content)
        if "addendum" in fn or "addendum" in content[:300]:
            add_num = int(addendum_match.group(1)) if addendum_match else 1
            return "addendum", add_num

        if "affidavit" in fn or "affidavit" in content[:300]:
            return "affidavit", None

        if "spec" in fn or "specification" in fn:
            return "specs", None

        if fn.endswith(".html") or fn.endswith(".htm") or "bid information" in fn:
            return "bid_page", None

        if "rfp" in fn or "porfp" in fn or "proposal" in fn or "request" in fn:
            return "rfp", None

        # Default fallback based on content
        if "purchase order request for proposals" in content or "request for proposal" in content:
            return "rfp", None

        return "rfp", None

    @staticmethod
    def extract_document_date(text: str) -> Optional[str]:
        """Extracts primary document date if present in headers or intro text."""
        date_patterns = [
            r"(?:Date|Issue Date|Publication|Published|Closing Date)[:\s]+([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})",
            r"(?:Date|Issue Date|Publication|Published)[:\s]+([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
            r"([0-9]{2}-[A-Za-z]{3}-[0-9]{4})"
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text[:2000], re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text:
        - Removes repeated headers/footers
        - Fixes broken lines and hyphenation across lines (e.g., 'solici-\ntation' -> 'solicitation')
        - Normalizes multiple spaces and tabs
        """
        if not text:
            return ""

        # Remove form feed and unusual control chars
        text = text.replace("\x0c", "\n").replace("\r", "\n")

        # Fix hyphenated line breaks
        text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)

        # Remove repeated boilerplates or page markers (e.g., 'Page 1 | 5')
        text = re.sub(r"(?i)\bPage\s+\d+\s*(?:of|/|\|)\s*\d+\b", "", text)

        # Normalize multiple spaces into single space on same line
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def parse_pdf(self, file_path: str, bid_id: str) -> List[ParsedPage]:
        """
        Parses PDF file page-by-page with table extraction and fallback handling.
        """
        file_name = os.path.basename(file_path)
        pages: List[ParsedPage] = []

        # First preview doc type
        doc_type, addendum_num = self.detect_doc_type(file_name)

        try:
            # Use pdfplumber for table extraction & layout-aware text
            with pdfplumber.open(file_path) as pdf:
                doc_date = None
                for idx, page in enumerate(pdf.pages):
                    page_num = idx + 1
                    raw_text = page.extract_text() or ""

                    # Extract tables if present
                    tables_md = []
                    try:
                        extracted_tables = page.extract_tables()
                        for tbl in extracted_tables:
                            if tbl and len(tbl) > 1:
                                # Convert table rows to markdown table
                                header = [str(c or "").strip().replace("\n", " ") for c in tbl[0]]
                                md_rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
                                for row in tbl[1:]:
                                    row_vals = [str(c or "").strip().replace("\n", " ") for c in row]
                                    # Ensure column count match
                                    if len(row_vals) < len(header):
                                        row_vals += [""] * (len(header) - len(row_vals))
                                    md_rows.append("| " + " | ".join(row_vals[:len(header)]) + " |")
                                tables_md.append("\n".join(md_rows))
                    except Exception as e:
                        logger.debug(f"Table extraction note on {file_name} p.{page_num}: {e}")

                    cleaned_text = self.clean_text(raw_text)

                    # Date detection from early pages
                    if not doc_date and page_num <= 3:
                        doc_date = self.extract_document_date(raw_text)

                    if not doc_type or doc_type == "rfp":
                        detected_t, detected_add = self.detect_doc_type(file_name, raw_text)
                        if detected_t:
                            doc_type = detected_t
                        if detected_add:
                            addendum_num = detected_add

                    # Build combined text representation
                    content_parts = []
                    if cleaned_text:
                        content_parts.append(cleaned_text)
                    if tables_md:
                        content_parts.append("\n\n### Extracted Tables:\n" + "\n\n".join(tables_md))

                    final_content = "\n\n".join(content_parts)

                    if not final_content.strip():
                        logger.warning(f"Empty page encountered: {file_name} Page {page_num}")
                        continue

                    pages.append(ParsedPage(
                        bid_id=bid_id,
                        file_name=file_name,
                        page_number=page_num,
                        doc_type=doc_type,
                        addendum_number=addendum_num,
                        document_date=doc_date,
                        text=final_content,
                        tables=tables_md,
                        metadata={
                            "char_count": len(final_content),
                            "table_count": len(tables_md),
                            "file_path": file_path
                        }
                    ))

        except Exception as e:
            logger.error(f"Error parsing PDF with pdfplumber ({file_path}): {e}. Falling back to pypdf.")
            # Fallback to pypdf
            try:
                reader = PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    page_num = idx + 1
                    raw_text = page.extract_text() or ""
                    cleaned = self.clean_text(raw_text)
                    if cleaned:
                        pages.append(ParsedPage(
                            bid_id=bid_id,
                            file_name=file_name,
                            page_number=page_num,
                            doc_type=doc_type,
                            addendum_number=addendum_num,
                            document_date=None,
                            text=cleaned,
                            tables=[],
                            metadata={"fallback": True, "file_path": file_path}
                        ))
            except Exception as e2:
                logger.error(f"Fatal error reading PDF {file_path}: {e2}")

        logger.info(f"Parsed {file_name}: {len(pages)} pages (Type: {doc_type}, Addendum: {addendum_num})")
        return pages

    def parse_html(self, file_path: str, bid_id: str) -> List[ParsedPage]:
        """
        Parses HTML bid page (e.g. BidNet Direct portal page) extracting metadata,
        structured definition tables, and cleaned text while filtering out nav/scripts.
        """
        file_name = os.path.basename(file_path)
        pages: List[ParsedPage] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove noise scripts, styles, navigations, footers
            for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
                tag.decompose()

            # Structured key-value extraction for Bid portals
            structured_sections = []
            title = soup.title.string.strip() if soup.title and soup.title.string else file_name
            structured_sections.append(f"Portal Page Title: {title}")

            # Extract tables or definition list items
            tables_md = []
            for table in soup.find_all("table"):
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
                    if cells:
                        rows.append(cells)
                if len(rows) > 1:
                    max_cols = max(len(r) for r in rows)
                    hdr = rows[0] + [""] * (max_cols - len(rows[0]))
                    md = ["| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * max_cols) + " |"]
                    for r in rows[1:]:
                        r = r + [""] * (max_cols - len(r))
                        md.append("| " + " | ".join(r) + " |")
                    tables_md.append("\n".join(md))

            # Extract clean linear text from paragraphs and content divs
            text_chunks = []
            for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "li", "span"]):
                t = elem.get_text(" ", strip=True)
                if t and len(t) > 3:
                    text_chunks.append(t)

            # Combine and deduplicate
            raw_text = "\n".join(text_chunks)
            cleaned_text = self.clean_text(raw_text)

            # Look for publication / closing dates
            doc_date = self.extract_document_date(cleaned_text)

            pages.append(ParsedPage(
                bid_id=bid_id,
                file_name=file_name,
                page_number=1,
                doc_type="bid_page",
                addendum_number=None,
                document_date=doc_date,
                text=cleaned_text + ("\n\n### Extracted Tables:\n" + "\n\n".join(tables_md) if tables_md else ""),
                tables=tables_md,
                metadata={
                    "title": title,
                    "table_count": len(tables_md),
                    "file_path": file_path
                }
            ))
            logger.info(f"Parsed HTML {file_name}: 1 page extracted.")

        except Exception as e:
            logger.error(f"Error parsing HTML {file_path}: {e}")

        return pages

    def parse_bid_directory(self, dir_path: str, bid_id: Optional[str] = None) -> List[ParsedPage]:
        """
        Parses all supported documents (PDF, HTML) inside a bid directory.
        """
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        bid_id = bid_id or os.path.basename(dir_path.rstrip("/"))
        all_pages: List[ParsedPage] = []

        for fname in sorted(os.listdir(dir_path)):
            if fname.startswith("."):
                continue
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext == ".pdf":
                pages = self.parse_pdf(fpath, bid_id)
                all_pages.extend(pages)
            elif ext in [".html", ".htm"]:
                pages = self.parse_html(fpath, bid_id)
                all_pages.extend(pages)
            else:
                logger.info(f"Skipping unsupported file type: {fname}")

        logger.info(f"Total parsed pages for {bid_id}: {len(all_pages)}")
        return all_pages
