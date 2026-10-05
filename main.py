"""
CLI & Service Entry Point for RFP Intelligence Platform
Supports extraction mode, Q&A mode, search mode, evaluation mode, API server, and Streamlit UI.
"""

import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.orchestrator import Orchestrator
from search.indexer import SearchIndexer
from search.hybrid_retriever import HybridRetriever
from eval.evaluate_retrieval import run_full_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rfp_platform.main")


def main():
    parser = argparse.ArgumentParser(
        description="RFP Intelligence Platform: RAG Search Engine & Multi-Agent System"
    )
    parser.add_argument("--bid", type=str, help="Path to bid directory for extraction (e.g. ./data/Bid1)")
    parser.add_argument("--bid-id", type=str, default=None, help="Explicit bid ID override (e.g. Bid1)")
    parser.add_argument("--ask", type=str, help="Free-form natural language question to answer with citations")
    parser.add_argument("--search", type=str, help="Execute hybrid search query on indexed documents")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved results (default: 5)")
    parser.add_argument("--eval", action="store_true", help="Run the 4-configuration retrieval evaluation benchmark")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI REST API server")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit interactive Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server (default: 8000)")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path for extraction")
    parser.add_argument("--compare", action="store_true", help="Generate side-by-side comparison report of extracted bids")
    parser.add_argument("--gonogo", type=str, default=None, help="Run automated Go / No-Go pursuit recommendation on a bid (e.g. Bid1, Bid2)")

    args = parser.parse_args()

    # Mode 1: Retrieval Evaluation Benchmark
    if args.eval:
        print("\n" + "=" * 70)
        print("          RUNNING SECTION 6.4 RETRIEVAL EVALUATION BENCHMARK")
        print("=" * 70)
        report = run_full_evaluation()
        print("\n" + "=" * 78)
        print("                    RETRIEVAL EVALUATION BENCHMARK RESULTS")
        print("=" * 78)
        print(f"{'Configuration':<35} | {'Recall@1':<8} | {'Recall@3':<8} | {'Recall@5':<8} | {'MRR':<6} | {'nDCG@5':<6}")
        print("-" * 78)
        for c in report["configurations"]:
            print(f"{c['Configuration']:<35} | {c['Recall@1']:<8.4f} | {c['Recall@3']:<8.4f} | {c['Recall@5']:<8.4f} | {c['MRR']:<6.4f} | {c['nDCG@5']:<6.4f}")
        print("=" * 78)
        print(f"\nReport written to: ./output/retrieval_eval_report.json\n")
        return

    # Mode 2: FastAPI Server
    if args.api:
        import uvicorn
        print(f"Starting RFP Intelligence FastAPI Server on http://0.0.0.0:{args.port}...")
        uvicorn.run("api.app:app", host="0.0.0.0", port=args.port, reload=False)
        return

    # Mode 3: Streamlit UI
    if args.ui:
        import subprocess
        print("Launching Streamlit Web UI on http://localhost:8501...")
        cmd = [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"]
        subprocess.run(cmd)
        return

    # Mode: Bid Comparison Report
    if args.compare:
        from agents.comparison_agent import BidComparisonAgent
        with open("output/bid1_extracted.json") as f:
            b1 = json.load(f)
        with open("output/bid2_extracted.json") as f:
            b2 = json.load(f)
        comparator = BidComparisonAgent()
        report = comparator.compare_bids([b1, b2])
        out_comp = "output/bid_comparison_report.json"
        with open(out_comp, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        print("\n" + "=" * 70)
        print("                 SIDE-BY-SIDE RFP COMPARISON MATRIX")
        print("=" * 70)
        for row in report.matrix:
            print(f"\n* {row['Dimension']}:")
            for b_id in report.bids_compared:
                print(f"    - {b_id}: {row.get(b_id, 'N/A')}")
        print("\n" + "=" * 70)
        print(f"Executive Synthesis:\n{report.executive_synthesis}\n")
        print(f"Report saved to: {out_comp}\n")
        return

    # Mode: Go / No-Go Decision
    if args.gonogo:
        from agents.gonogo_agent import GoNoGoAgent
        target_bid = args.gonogo.lower()
        target_file = f"output/{target_bid}_extracted.json" if not target_bid.endswith(".json") else target_bid
        if not os.path.exists(target_file):
            print(f"Error: Extracted bid file '{target_file}' not found. Run --bid first.")
            return
        with open(target_file) as f:
            b_data = json.load(f)
        agent = GoNoGoAgent()
        rec = agent.evaluate_bid(b_data)
        print("\n" + "=" * 70)
        print(f"       AUTOMATED PURSUIT DECISION: {rec.decision} ({rec.bid_id})")
        print("=" * 70)
        print(f"Pursuit Match Score: {rec.match_score}/100.0")
        print(f"\nExecutive Summary:\n{rec.executive_summary}\n")
        print("Operational Strengths:")
        for s in rec.strengths:
            print(f"  + {s}")
        if rec.risks_and_blockers:
            print("\nBlockers / Action Items:")
            for r in rec.risks_and_blockers:
                print(f"  - {r}")
        print("=" * 70 + "\n")
        return

    # Initialize Orchestrator for Extraction, Q&A, and Search
    indexer = SearchIndexer()
    retriever = HybridRetriever(indexer=indexer)
    orchestrator = Orchestrator(indexer=indexer, retriever=retriever)

    # Mode 4: Free-form Q&A
    if args.ask:
        print(f"\n--- Question: {args.ask} ---")
        resp = orchestrator.answer_question(args.ask, bid_id=args.bid_id)
        print(f"\nAnswer:\n{resp.answer}\n")
        print("Citations:")
        for idx, c in enumerate(resp.citations, 1):
            print(f"  [{idx}] {c.file} (Page {c.page}) - Score: {c.score or 0.0:.4f}")
            print(f"      \"{c.text_snippet}\"")
        return

    # Mode 5: Hybrid Search Query
    if args.search:
        print(f"\n--- Search Query: '{args.search}' (Bid Filter: {args.bid_id or 'All'}) ---")
        results = retriever.search(
            query=args.search,
            bid_id=args.bid_id,
            top_k=args.top_k,
            mode="hybrid",
            apply_rerank=True,
            apply_expansion=True
        )
        for idx, r in enumerate(results, 1):
            print(f"\n#{idx} | [{r.bid_id}] {r.file_name} (Page {r.page_number}) | DocType: {r.doc_type} | Score: {r.score:.4f}")
            print(f"Snippet: {r.text[:250].strip()}...")
        return

    # Mode 6: Structured Bid Extraction
    if args.bid:
        bid_folder = args.bid
        bid_id = args.bid_id or os.path.basename(bid_folder.rstrip("/"))
        print(f"\n>>> Running Multi-Agent Extraction Pipeline on: {bid_folder} (ID: {bid_id})")
        state = orchestrator.run_extraction_pipeline(bid_folder, bid_id)

        out_path = args.output or f"output/{bid_id.lower()}_extracted.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(state.final_output, f, indent=2)

        print("\n" + "=" * 60)
        print(f"       EXTRACTION COMPLETED FOR {bid_id}")
        print("=" * 60)
        val = state.validation_summary
        print(f"Validation Results: {val.passed} Passed | {val.failed} Failed | {val.not_found} Not Found / Null (Total: {val.total})")
        print(f"Addendum Changes Reconciled: {len(state.addendum_changes)}")
        print(f"Output saved to: {out_path}")
        print("=" * 60 + "\n")
        return

    # If no flags provided, display help
    parser.print_help()


if __name__ == "__main__":
    main()
