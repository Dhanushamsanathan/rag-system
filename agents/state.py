"""
Shared State and Memory Models for RFP Multi-Agent System
Defines Pydantic models for agent communication, state persistence, citations, and tracing.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source evidence citation backing an extracted field or answer."""
    file: str
    page: int
    text_snippet: Optional[str] = None
    score: Optional[float] = None


class FieldResult(BaseModel):
    """An individual extracted RFP field with value, sources, confidence, and notes."""
    field_name: str
    value: Any = None
    sources: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    validated: bool = False
    validation_notes: Optional[str] = None
    retry_count: int = 0


class AddendumChange(BaseModel):
    """Represents a specific modification introduced by an addendum."""
    field_name: str
    original_value: Any = None
    new_value: Any = None
    addendum_file: str
    page: int
    reason: str


class TraceStep(BaseModel):
    """Observability trace event for an agent invocation."""
    step_id: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    agent_name: str
    action: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    output_data: Any = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    status: str = "success"


class ValidationSummary(BaseModel):
    """Aggregated validation statistics."""
    passed: int = 0
    failed: int = 0
    not_found: int = 0
    total: int = 20


class RFPOutputSchema(BaseModel):
    """Standard Section 8.1 RFP Output JSON Schema."""
    bid_id: str
    fields: Dict[str, Dict[str, Any]]
    addendum_changes: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Dict[str, int] = Field(default_factory=dict)


class SharedAgentState(BaseModel):
    """
    Central shared state passed between agents in the pipeline.
    Preserves plan, retrieved evidence, field drafts, addendum changes, and full execution trace.
    """
    bid_id: str
    bid_path: str
    plan: List[str] = Field(default_factory=list)
    current_step: int = 0
    retrieved_evidence: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    draft_fields: Dict[str, FieldResult] = Field(default_factory=dict)
    addendum_changes: List[AddendumChange] = Field(default_factory=list)
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    trace: List[TraceStep] = Field(default_factory=list)
    completed: bool = False
    final_output: Optional[Dict[str, Any]] = None

    def add_trace(self, agent_name: str, action: str, input_data: Any, output_data: Any, tool_calls: Optional[List[Dict[str, Any]]] = None, latency_ms: float = 0.0, tokens: int = 0, status: str = "success"):
        step_id = len(self.trace) + 1
        self.trace.append(TraceStep(
            step_id=step_id,
            agent_name=agent_name,
            action=action,
            input_data={"data": input_data} if not isinstance(input_data, dict) else input_data,
            tool_calls=tool_calls or [],
            output_data=output_data,
            latency_ms=round(latency_ms, 2),
            tokens_used=tokens,
            status=status
        ))
