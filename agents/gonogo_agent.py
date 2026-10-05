"""
Go / No-Go Decision Recommendation Agent Module
Analyzes RFP bids against configurable company capability profiles to provide
an automated, justified GO / CONDITIONAL GO / NO-GO recommendation.
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("rfp_platform.gonogo_agent")


class CapabilityProfile(BaseModel):
    """Company operational profile against which bids are evaluated."""
    company_name: str = "Enterprise IT Solutions Corp"
    authorized_oems: List[str] = Field(default_factory=lambda: ["Dell", "Lenovo", "HP", "Apple"])
    supports_white_glove: bool = True
    supports_asset_tagging: bool = True
    min_delivery_window_days: int = 30
    registered_cooperatives: List[str] = Field(
        default_factory=lambda: ["EPCNT", "060B5400007", "Desktop, Laptop and Tablet 2015"]
    )
    small_business_certified: bool = True
    max_bonding_capacity_usd: float = 5_000_000.0


class GoNoGoRecommendation(BaseModel):
    """Structured output for Go/No-Go evaluation."""
    bid_id: str
    decision: str  # GO, CONDITIONAL_GO, NO_GO
    match_score: float  # 0.0 to 100.0
    strengths: List[str] = Field(default_factory=list)
    risks_and_blockers: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    executive_summary: str = ""


class GoNoGoAgent:
    """Evaluates extracted RFP fields to recommend bid pursuit viability."""

    def __init__(self, profile: Optional[CapabilityProfile] = None):
        self.profile = profile or CapabilityProfile()

    def evaluate_bid(self, bid_data: Dict[str, Any]) -> GoNoGoRecommendation:
        """
        Conducts compliance evaluation across 5 operational pillars:
        1. Manufacturer Partner Authorization
        2. Deployment & Installation Scope
        3. Contract Vehicle & Set-Aside Eligibility
        4. Delivery Timelines
        5. Bonding & Legal Requirements
        """
        bid_id = bid_data.get("bid_id", "Unknown")
        fields = bid_data.get("fields", {})

        score = 0.0
        strengths = []
        risks = []
        mitigations = []

        # 1. Manufacturer Authorization (Weight: 25 pts)
        mfg = fields.get("MFG for Registration", {}).get("value", "") or ""
        mfg_lower = mfg.lower()
        matched_oem = any(oem.lower() in mfg_lower for oem in self.profile.authorized_oems)
        if matched_oem:
            score += 25.0
            strengths.append(f"OEM Authorization: Fully certified with requested manufacturer partner ({mfg[:60]}).")
        else:
            risks.append("OEM Authorization: Proposed hardware may require vendor partner sub-authorization.")

        # 2. Scope & Installation (Weight: 20 pts)
        installation = fields.get("Installation", {}).get("value", "") or ""
        if "white glove" in installation.lower():
            if self.profile.supports_white_glove:
                score += 20.0
                strengths.append("Installation Capability: White Glove staging and asset tagging are fully supported internally.")
            else:
                score += 5.0
                risks.append("White Glove services required but not supported in-house; requires third-party logistics partner.")
                mitigations.append("Subcontract asset tagging and etching to local field service partner.")
        else:
            score += 20.0
            strengths.append("Installation Scope: Hardware delivery only; low operational staging complexity.")

        # 3. Contract Vehicle & Set-Aside (Weight: 20 pts)
        coop = fields.get("Contract or Cooperative to use", {}).get("value", "") or ""
        coop_matched = any(c.lower() in coop.lower() for c in self.profile.registered_cooperatives)
        if coop_matched or not coop:
            score += 20.0
            strengths.append("Procurement Vehicle: Eligible under registered state/cooperative contract vehicle.")
        else:
            score += 10.0
            risks.append(f"Referenced contract vehicle ({coop[:50]}) requires active membership verification.")

        # 4. Delivery & Terms (Weight: 20 pts)
        delivery = fields.get("Delivery Date", {}).get("value", "") or ""
        if "45 days" in delivery or "varied" in delivery:
            score += 20.0
            strengths.append("Delivery Timeline: 45-day window aligns comfortably with OEM supply chain lead times.")
        else:
            score += 15.0
            strengths.append("Standard delivery conditions applicable.")

        # 5. Bonding & Additional Docs (Weight: 15 pts)
        bond = fields.get("Bid Bond Requirement", {}).get("value")
        if bond is None or "not required" in str(fields.get("Bid Bond Requirement", {}).get("notes", "")).lower():
            score += 15.0
            strengths.append("Financial Bonding: Zero bid bond requirement eliminates upfront surety overhead.")
        else:
            score += 10.0
            risks.append("Bid bond required; requires treasury underwriting.")

        # Decision Determination
        if score >= 80.0 and len(risks) == 0:
            decision = "GO"
            summary = (
                f"Definitive GO recommendation for {bid_id} (Score: {score:.1f}/100). The RFP aligns seamlessly "
                f"with company OEM authorizations, internal staging capabilities, and contract vehicles."
            )
        elif score >= 65.0:
            decision = "CONDITIONAL_GO"
            summary = (
                f"CONDITIONAL GO recommendation for {bid_id} (Score: {score:.1f}/100). High probability of win, "
                f"subject to addressing identified items: {'; '.join(risks)}."
            )
        else:
            decision = "NO_GO"
            summary = f"NO-GO recommendation for {bid_id} (Score: {score:.1f}/100) due to significant alignment risks."

        return GoNoGoRecommendation(
            bid_id=bid_id,
            decision=decision,
            match_score=round(score, 1),
            strengths=strengths,
            risks_and_blockers=risks,
            mitigation_strategies=mitigations,
            executive_summary=summary
        )
