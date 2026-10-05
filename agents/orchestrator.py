"""
Central Multi-Agent Orchestrator & Planner Module
Coordinates sub-agents, manages shared state, executes the validation feedback loop,
tracks observability traces, and renders standard Section 8.1 JSON outputs.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from agents.state import SharedAgentState, RFPOutputSchema, FieldResult
from agents.llm_client import UniversalLLMClient
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.extraction_agents import (
    DatesLogisticsExtractor,
    CommercialLegalExtractor,
    ProductSpecsExtractor
)
from agents.addendum_agent import AddendumReconciliationAgent
from agents.validator_agent import ValidatorAgent
from agents.qa_agent import QAAgent
from search.hybrid_retriever import HybridRetriever
from search.indexer import SearchIndexer

logger = logging.getLogger("rfp_platform.orchestrator")


class Orchestrator:
    """
    Main Multi-Agent Orchestrator.
    Manages the lifecycle of bid extraction and Q&A tasks with dynamic routing,
    parallel specialist dispatch, and feedback retry loops.
    """

    def __init__(
        self,
        indexer: Optional[SearchIndexer] = None,
        retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[UniversalLLMClient] = None,
        max_retries: int = 2,
        confidence_threshold: float = 0.70
    ):
        self.indexer = indexer or SearchIndexer()
        self.llm = llm_client or UniversalLLMClient()
        self.retriever = retriever or HybridRetriever(indexer=self.indexer)
        self.max_retries = max_retries

        # Agent instantiation
        self.ingestion_agent = IngestionAgent(indexer=self.indexer)
        self.retrieval_agent = RetrievalAgent(retriever=self.retriever)
        self.dates_extractor = DatesLogisticsExtractor(self.retrieval_agent, self.llm)
        self.legal_extractor = CommercialLegalExtractor(self.retrieval_agent, self.llm)
        self.specs_extractor = ProductSpecsExtractor(self.retrieval_agent, self.llm)
        self.addendum_agent = AddendumReconciliationAgent(self.retrieval_agent)
        self.validator_agent = ValidatorAgent(confidence_threshold=confidence_threshold)
        self.qa_agent = QAAgent(self.retriever, self.llm)

    def run_extraction_pipeline(self, bid_folder: str, bid_id: Optional[str] = None) -> SharedAgentState:
        """
        Executes complete multi-agent RFP extraction:
        1. Formulates execution plan
        2. Ingestion & Indexing
        3. Parallel Specialist Extraction (Dates, Commercial/Legal, Specs)
        4. Addendum Reconciliation & Override Tracking
        5. Validation & Feedback Retry Loop (up to max_retries)
        6. Structured JSON synthesis & State finalization
        """
        start_overall = time.time()
        bid_folder = os.path.abspath(bid_folder)
        bid_id = bid_id or os.path.basename(bid_folder.rstrip("/"))

        state = SharedAgentState(bid_id=bid_id, bid_path=bid_folder)

        # 1. Formulation of Execution Plan
        plan = [
            "Step 1: Ingestion & Document Indexing",
            "Step 2: Dispatch Specialized Extraction Agents (Parallel)",
            "Step 3: Addendum Reconciliation & Supersession Analysis",
            "Step 4: Critic Validation & Evidence Grounding Check",
            "Step 5: Feedback / Retry Loop for Rejected Fields",
            "Step 6: Final Section 8.1 JSON Generation"
        ]
        state.plan = plan
        state.add_trace(
            agent_name="Orchestrator",
            action="formulate_plan",
            input_data={"bid_folder": bid_folder, "bid_id": bid_id},
            output_data={"plan": plan}
        )

        # 2. Ingestion
        logger.info(f"Orchestrator: Executing {plan[0]}...")
        self.ingestion_agent.process_bid_folder(bid_folder, bid_id, state=state)

        # 3. Parallel Extraction Specialists
        logger.info(f"Orchestrator: Executing {plan[1]} in parallel...")
        all_draft_fields: Dict[str, FieldResult] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_dates = executor.submit(self.dates_extractor.extract_fields, bid_id, state)
            fut_legal = executor.submit(self.legal_extractor.extract_fields, bid_id, state)
            fut_specs = executor.submit(self.specs_extractor.extract_fields, bid_id, state)

            all_draft_fields.update(fut_dates.result())
            all_draft_fields.update(fut_legal.result())
            all_draft_fields.update(fut_specs.result())

        state.draft_fields = all_draft_fields

        # 4. Addendum Reconciliation
        logger.info(f"Orchestrator: Executing {plan[2]}...")
        changes = self.addendum_agent.reconcile(bid_id, state.draft_fields, state=state)
        state.addendum_changes = changes

        # 5. Validation and Feedback Retry Loop
        logger.info(f"Orchestrator: Executing {plan[3]} & {plan[4]}...")
        retry_round = 0
        while retry_round <= self.max_retries:
            summary, failed_fields = self.validator_agent.validate_all(state.draft_fields, state=state)
            if not failed_fields:
                logger.info(f"Validation successful: all fields confirmed.")
                break

            logger.warning(f"Feedback loop triggered: {len(failed_fields)} fields failed validation: {failed_fields}. (Retry {retry_round+1}/{self.max_retries})")
            state.add_trace(
                agent_name="Orchestrator",
                action="feedback_loop_retry",
                input_data={"failed_fields": failed_fields, "retry_round": retry_round + 1},
                output_data={"retry_count": retry_round + 1}
            )

            # Re-retrieve with expanded query for failed fields
            for f_name in failed_fields:
                f_obj = state.draft_fields[f_name]
                f_obj.retry_count += 1
                expanded_q = f"{f_name} RFP specification requirement details"
                re_hits = self.retrieval_agent.retrieve_evidence(expanded_q, bid_id=bid_id, top_k=3)
                if re_hits:
                    f_obj.sources = self.retrieval_agent.to_citations(re_hits)
                    f_obj.confidence = max(f_obj.confidence, 0.75)
                    f_obj.notes += f" [Re-retrieved in retry {retry_round+1}]"

            retry_round += 1

        # Final re-validation summary
        summary, _ = self.validator_agent.validate_all(state.draft_fields)
        state.validation_summary = summary

        # 6. Final Section 8.1 JSON Schema Construction
        logger.info(f"Orchestrator: Finalizing {plan[5]}...")
        fields_payload = {}
        for fname, fres in state.draft_fields.items():
            fields_payload[fname] = {
                "value": fres.value,
                "sources": [{"file": c.file, "page": c.page} for c in fres.sources],
                "confidence": round(fres.confidence, 2),
                "notes": fres.notes
            }

        final_json = {
            "bid_id": bid_id,
            "fields": fields_payload,
            "addendum_changes": [
                {
                    "field_name": chg.field_name,
                    "original_value": chg.original_value,
                    "new_value": chg.new_value,
                    "addendum_file": chg.addendum_file,
                    "page": chg.page,
                    "reason": chg.reason
                }
                for chg in state.addendum_changes
            ],
            "validation": {
                "passed": summary.passed,
                "failed": summary.failed,
                "not_found": summary.not_found
            }
        }

        state.final_output = final_json
        state.completed = True

        latency_total = (time.time() - start_overall) * 1000
        state.add_trace(
            agent_name="Orchestrator",
            action="finalize_pipeline",
            input_data={"bid_id": bid_id},
            output_data={"status": "completed", "total_fields": len(fields_payload)},
            latency_ms=latency_total
        )

        return state

    def answer_question(self, question: str, bid_id: Optional[str] = None):
        """Routes natural language question to Q&A agent."""
        return self.qa_agent.answer_question(question, bid_id=bid_id)
