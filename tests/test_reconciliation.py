"""
Unit Tests for Addendum Reconciliation and Validator Critic
"""

import pytest
from agents.state import FieldResult, Citation
from agents.validator_agent import ValidatorAgent


@pytest.fixture
def validator():
    return ValidatorAgent(confidence_threshold=0.70)


def test_validator_with_valid_field(validator):
    field = FieldResult(
        field_name="Due Date",
        value="2024-07-09 14:00 CST",
        sources=[Citation(file="Addendum 2.pdf", page=1, text_snippet="The new due date for this RFP will be July 9, 2024 at 2:00 PM CST.")],
        confidence=0.95,
        notes="Extended by Addendum 2"
    )
    is_valid, msg = validator.validate_field(field)
    assert is_valid
    assert field.validated


def test_validator_detects_missing_citation(validator):
    field = FieldResult(
        field_name="Title",
        value="Invented RFP Title",
        sources=[],
        confidence=0.90
    )
    is_valid, msg = validator.validate_field(field)
    assert not is_valid
    assert "Missing required source citation" in msg


def test_validator_valid_null_field(validator):
    field = FieldResult(
        field_name="Bid Bond Requirement",
        value=None,
        sources=[],
        confidence=0.85,
        notes="Not found in documents"
    )
    is_valid, msg = validator.validate_field(field)
    assert is_valid
