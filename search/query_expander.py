"""
Query Understanding & Expansion Module
Expands user and agent queries with domain-specific procurement terminology and synonyms.
"""

import re
import logging
from typing import List

logger = logging.getLogger("rfp_platform.query_expander")

DOMAIN_SYNONYMS = {
    "deadline": ["due date", "submission deadline", "closing date", "solicitation due", "proposal due date"],
    "due date": ["submission deadline", "closing date", "solicitation due", "new due date", "extended due date"],
    "bond": ["bid bond", "proposal bond", "bid security", "surety bond", "bonding requirements"],
    "warranty": ["extended warranty", "manufacturer warranty", "limited hardware warranty", "warranty service", "3-year warranty"],
    "specs": ["product specifications", "technical requirements", "hardware configuration", "CPU", "RAM", "storage", "display"],
    "specification": ["product specifications", "technical requirements", "hardware configuration", "processor", "memory"],
    "contact": ["procurement contact", "buyer", "point of contact", "POC", "purchasing department", "email", "phone"],
    "term": ["contract term", "duration of contract", "renewal options", "period of performance", "3 year agreement"],
    "meeting": ["pre-bid meeting", "pre-proposal conference", "mandatory attendance", "pre-bidding events"],
    "pre bid": ["pre-proposal meeting", "pre-proposal conference", "prebid event"],
    "affidavit": ["mercury affidavit", "contract affidavit", "interested parties", "form 1295", "sworn statement"],
    "delivery": ["delivery date", "lead time", "shipping terms", "delivery window", "days after award"],
    "payment": ["payment terms", "invoicing instructions", "Net 30", "accounts payable", "prompt payment"],
    "cooperative": ["interlocal agreement", "EPCNT", "purchasing cooperative", "piggyback contract", "master contract"],
    "manufacturer": ["MFG registration", "authorized reseller", "OEM", "letter of authorization", "partner tier"],
    "addendum": ["addendum changes", "addendum 1", "addendum 2", "amendment", "clarification questions"]
}


class QueryExpander:
    """Expands queries with procurement-specific synonyms and contextual keywords."""

    def __init__(self, domain_dict=None):
        self.synonyms = domain_dict or DOMAIN_SYNONYMS

    def expand(self, query: str) -> List[str]:
        """
        Takes a raw query and returns a list of terms/variations:
        [original_query, expanded_combination_query, sub-terms...]
        """
        query_clean = query.strip()
        matched_expansions = []

        q_lower = query_clean.lower()
        for key, syns in self.synonyms.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q_lower):
                matched_expansions.extend(syns[:3])

        if not matched_expansions:
            return [query_clean]

        # Combine unique expansions
        combined = f"{query_clean} " + " ".join(matched_expansions[:4])
        return [query_clean, combined]
