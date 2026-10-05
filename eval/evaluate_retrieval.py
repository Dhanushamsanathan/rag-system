"""
Retrieval Evaluation Engine
Evaluates retrieval performance across Vector-Only, BM25-Only, Hybrid RRF, and Hybrid + Re-ranker.
Calculates Recall@1, Recall@3, Recall@5, MRR, and nDCG@5 on the 18-question benchmark dataset.
"""

import os
import json
import math
import logging
from typing import Dict, Any, List

from search.indexer import SearchIndexer
from search.hybrid_retriever import HybridRetriever
from eval.benchmark_data import EVALUATION_PAIRS

logger = logging.getLogger("rfp_platform.eval")


def compute_ndcg_at_k(relevant_ranks: List[int], k: int = 5) -> float:
    """Computes Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for rank in relevant_ranks:
        if rank <= k:
            dcg += 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)  # Ideal DCG for 1 primary target
    return min(1.0, dcg / idcg)


def is_relevant(hit, pair: Dict[str, Any]) -> bool:
    """Checks whether a retrieved hit matches ground truth file, page, or keywords."""
    file_match = hit.file_name in pair["expected_files"]
    page_match = hit.page_number in pair["expected_pages"]
    text_lower = hit.text.lower()
    keyword_match = any(k.lower() in text_lower for k in pair["keywords"])

    # Consider relevant if file+page match, or file+keyword match
    if file_match and (page_match or keyword_match):
        return True
    return False


def evaluate_configuration(retriever: HybridRetriever, config_name: str, mode: str, apply_rerank: bool) -> Dict[str, float]:
    """Runs all benchmark queries through the specified retrieval configuration."""
    recall_1_count = 0
    recall_3_count = 0
    recall_5_count = 0
    reciprocal_ranks = []
    ndcg_scores = []

    for pair in EVALUATION_PAIRS:
        q = pair["question"]
        bid_id = pair["bid_id"]

        hits = retriever.search(
            query=q,
            bid_id=bid_id,
            top_k=5,
            mode=mode,
            apply_rerank=apply_rerank,
            apply_expansion=True
        )

        first_rel_rank = 0
        rel_ranks = []
        for idx, hit in enumerate(hits):
            rank = idx + 1
            if is_relevant(hit, pair):
                rel_ranks.append(rank)
                if first_rel_rank == 0:
                    first_rel_rank = rank

        # Update metrics
        if first_rel_rank == 1:
            recall_1_count += 1
        if 1 <= first_rel_rank <= 3:
            recall_3_count += 1
        if 1 <= first_rel_rank <= 5:
            recall_5_count += 1

        reciprocal_ranks.append(1.0 / first_rel_rank if first_rel_rank > 0 else 0.0)
        ndcg_scores.append(compute_ndcg_at_k(rel_ranks, k=5))

    total = len(EVALUATION_PAIRS)
    return {
        "Configuration": config_name,
        "Recall@1": round(recall_1_count / total, 4),
        "Recall@3": round(recall_3_count / total, 4),
        "Recall@5": round(recall_5_count / total, 4),
        "MRR": round(sum(reciprocal_ranks) / total, 4),
        "nDCG@5": round(sum(ndcg_scores) / total, 4)
    }


def run_full_evaluation(output_path: str = "./output/retrieval_eval_report.json") -> Dict[str, Any]:
    """Executes comparative evaluation across all 4 configurations and writes JSON report."""
    indexer = SearchIndexer()
    retriever = HybridRetriever(indexer=indexer)

    configs = [
        ("Vector-Only (Dense BGE-Small)", "dense", False),
        ("Keyword-Only (BM25Okapi)", "sparse", False),
        ("Hybrid (Dense + BM25 RRF)", "hybrid", False),
        ("Hybrid + Cross-Encoder Re-ranker", "hybrid", True)
    ]

    results = []
    for name, mode, rerank in configs:
        logger.info(f"Evaluating {name}...")
        metrics = evaluate_configuration(retriever, name, mode, rerank)
        results.append(metrics)

    report = {
        "evaluation_pairs_count": len(EVALUATION_PAIRS),
        "configurations": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Retrieval evaluation report written to {output_path}")
    return report


if __name__ == "__main__":
    report = run_full_evaluation()
    print("\n" + "=" * 78)
    print("                    RETRIEVAL EVALUATION BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'Configuration':<35} | {'Recall@1':<8} | {'Recall@3':<8} | {'Recall@5':<8} | {'MRR':<6} | {'nDCG@5':<6}")
    print("-" * 78)
    for c in report["configurations"]:
        print(f"{c['Configuration']:<35} | {c['Recall@1']:<8.4f} | {c['Recall@3']:<8.4f} | {c['Recall@5']:<8.4f} | {c['MRR']:<6.4f} | {c['nDCG@5']:<6.4f}")
    print("=" * 78)
