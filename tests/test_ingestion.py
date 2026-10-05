"""
Unit Tests for Document Parsing and Ingestion
"""

import os
import pytest
from ingestion.document_parser import DocumentParser, ParsedPage


@pytest.fixture
def parser():
    return DocumentParser()


def test_detect_doc_type(parser):
    # Test addendum detection
    t1, num1 = parser.detect_doc_type("Addendum 1 RFP JA-207652.pdf")
    assert t1 == "addendum"
    assert num1 == 1

    t2, num2 = parser.detect_doc_type("Addendum No. 2.pdf")
    assert t2 == "addendum"
    assert num2 == 2

    # Test affidavit detection
    t3, num3 = parser.detect_doc_type("Mercury_Affidavit.pdf")
    assert t3 == "affidavit"
    assert num3 is None

    # Test specs detection
    t4, num4 = parser.detect_doc_type("Dell_Laptop_Specs.pdf")
    assert t4 == "specs"

    # Test HTML portal detection
    t5, num5 = parser.detect_doc_type("Bid Information.html")
    assert t5 == "bid_page"


def test_clean_text(parser):
    raw = "The solici-\ntation deadline is ex-\ntended. Page 1 of 5\nAll terms   remain  intact."
    cleaned = parser.clean_text(raw)
    assert "solicitation" in cleaned
    assert "extended" in cleaned
    assert "Page 1 of 5" not in cleaned
    assert "All terms remain intact." in cleaned


def test_parse_html(parser):
    html_file = "data/Bid2/Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html"
    if os.path.exists(html_file):
        pages = parser.parse_html(html_file, "Bid2")
        assert len(pages) == 1
        assert pages[0].doc_type == "bid_page"
        assert "Maryland" in pages[0].text or "Treasurer" in pages[0].text
