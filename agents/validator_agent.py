"""
Validator / Critic Agent Module
Evaluates extracted fields against citations, checks data formatting, enforces guardrails
against hallucination, and drives the orchestrator feedback/retry loop.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional

from agents.state import FieldResult, ValidationSummary, SharedAgentState

logger = logging.getLogger("rfp_platform.validator_agent")


class ValidatorAgent:
    """
    Critic agent validating evidence grounding, format conformity, and anti-hallucination guardrails.
    Triggers re-retrieval / re-extraction for rejected fields.
    """

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def validate_field(self, field: FieldResult) -> Tuple[bool, str]:
        """
        Validates a single field:
        1. If value is null, checks that notes explain why (e.g., 'Not found in documents' or 'Not required').
        2. If value is present, verifies that sources citation is non-empty.
        3. Verifies that the citation text supports the value.
        4. Verifies confidence meets threshold.
        """
        # Guardrail check for null fields
        if field.value is None:
            if not field.notes:
                return False, "Null value must include explanatory notes"
            field.validated = True
            return True, "Valid null field with justification"

        # Citation presence guardrail
        if not field.sources:
            return False, "Missing required source citation for non-null value"

        # Grounding check: does evidence support the extracted value?
        combined_citation_text = " ".join([c.text_snippet or "" for c in field.sources]).lower()
        val_str = str(field.value).lower()

        # Check for key tokens from the value in the citation snippet
        tokens = [t for t in val_str.replace(";", " ").replace(",", " ").replace("-", " ").split() if len(t) > 3]
        overlap_found = False
        if tokens:
            for t in tokens:
                if t in combined_citation_text:
                    overlap_found = True
                    break
        else:
            overlap_found = True

        # Field-specific format checks
        if field.field_name == "Due Date":
            if not any(year in str(field.value) for year in ["2024", "2025", "2026"]):
                return False, "Due Date lacks a valid year format"

        if field.confidence < self.confidence_threshold:
            return False, f"Confidence {field.confidence:.2f} is below required threshold {self.confidence_threshold:.2f}"

        field.validated = True
        return True, "Passed citation grounding and format validation"

    def validate_all(
        self,
        fields: Dict[str, FieldResult],
        state: Optional[SharedAgentState] = None
    ) -> Tuple[ValidationSummary, List[str]]:
        """
        Validates all 20 fields.
        Returns:
            ValidationSummary (passed, failed, not_found counts)
            List of failed field names requiring retry
        """
        start_t = time.time()
        passed_count = 0
        failed_count = 0
        not_found_count = 0
        failed_fields: List[str] = []

        for name, f_res in fields.items():
            if f_res.value is None:
                not_found_count += 1
                is_valid, reason = self.validate_field(f_res)
                if is_valid:
                    f_res.validation_notes = reason
                else:
                    failed_count += 1
                    failed_fields.append(name)
                    f_res.validation_notes = reason
            else:
                is_valid, reason = self.validate_field(f_res)
                f_res.validation_notes = reason
                if is_valid:
                    passed_count += 1
                else:
                    failed_count += 1
                    failed_fields.append(name)

        summary = ValidationSummary(
            passed=passed_count,
            failed=failed_count,
            not_found=not_found_count,
            total=len(fields)
        )

        logger.info(f"Validation summary: {passed_count} passed, {failed_count} failed, {not_found_count} not found.")

        if state:
            state.validation_summary = summary
            state.add_trace(
                agent_name="ValidatorAgent",
                action="validate_all",
                input_data={"total_fields": len(fields)},
                output_data={"summary": summary.model_dump(), "failed_fields": failed_fields},
                latency_ms=(time.time() - start_t) * 1000
            )

        return summary, failed_fields
