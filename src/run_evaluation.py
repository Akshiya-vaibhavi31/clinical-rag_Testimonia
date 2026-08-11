"""
run_evaluation.py

Phase 15: RAG Evaluation.

Runs every question in data/eval/eval_set.json through the actual pipeline
(rag_pipeline.answer_question) and computes real, reportable metrics:

RETRIEVAL METRICS:
  - Recall@k: for answerable questions, was at least one expected source
    actually present in the retrieved evidence?

REFUSAL METRICS (the core safety feature of this project):
  - Refusal precision: of the questions the system REFUSED, how many were
    actually supposed to be refused?
  - Refusal recall: of the questions that SHOULD have been refused, how many
    actually were? (low recall = answering things it shouldn't — the more
    dangerous failure mode in a medical context)

CITATION METRICS:
  - Fabrication rate: percentage of citations that were fabricated
  - Numeric mismatch rate: percentage of citations with unsupported numbers

Usage:
    python -m src.run_evaluation
"""

import json
import time

from src.config import PROJECT_ROOT
from src.rag_pipeline import answer_question
from src.evaluation_metrics import (
    compute_retrieval_ranking_metrics,
    compute_faithfulness,
    compute_answer_relevance,
    compute_citation_completeness,
    compute_safety_metrics,
)
from src.experiment_tracking import log_experiment, print_experiment_table

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "eval_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "eval" / "eval_checkpoint.json"


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_source_id(source_id: str) -> str:
    """Normalize whitespace so 'PMID  123' and 'PMID 123' compare equal."""
    return " ".join(source_id.split())


def get_source_label(chunk: dict) -> str:
    return chunk["source_id"] if chunk["source_type"] == "clinical_trial" else f"PMID {chunk['source_id']}"


def evaluate_single_question(eval_item: dict, embedding_model) -> dict:
    """Run one eval question through the real pipeline and score it fully."""
    result = answer_question(eval_item["question"], verbose=False)

    actual_refused = result["refused"]
    expected_refused = not eval_item["expected_answerable"]

    cited_ids = set(normalize_source_id(sid) for sid in result["cited_source_ids"])
    expected_ids = set(normalize_source_id(sid) for sid in eval_item["expected_source_ids"])

    # --- Retrieval ranking metrics (Recall@k, Precision@k, MRR, NDCG) ---
    # Computed on the PRE-RERANK candidate order (retrieval-stage quality)
    # and the FINAL post-rerank order (what actually reached the LLM), so we
    # can see whether reranking measurably improves ranking quality.
    retrieval_metrics_pre_rerank = None
    retrieval_metrics_post_rerank = None
    if eval_item["expected_answerable"] and expected_ids:
        pre_rerank_ids = [normalize_source_id(get_source_label(c)) for c in result["pre_rerank_candidates"]]
        post_rerank_ids = [normalize_source_id(get_source_label(c)) for c in result["evidence_chunks"]]
        retrieval_metrics_pre_rerank = compute_retrieval_ranking_metrics(pre_rerank_ids, expected_ids, k=5)
        retrieval_metrics_post_rerank = compute_retrieval_ranking_metrics(post_rerank_ids, expected_ids, k=5)

    verification = result["verification_report"] or {}

    # --- Generation quality metrics ---
    faithfulness = compute_faithfulness(verification) if verification else None
    answer_relevance = compute_answer_relevance(eval_item["question"], result["answer_text"], embedding_model)
    citation_completeness = compute_citation_completeness(result["answer_text"])

    # --- Safety metrics ---
    safety = (
        compute_safety_metrics(verification)
        if verification
        else {"unsupported_claim_rate": None, "hallucination_rate": None}
    )

    return {
        "id": eval_item["id"],
        "question_type": eval_item["question_type"],
        "question": eval_item["question"],
        "expected_answerable": eval_item["expected_answerable"],
        "actual_refused": actual_refused,
        "refusal_correct": actual_refused == expected_refused,
        "retrieval_metrics_pre_rerank": retrieval_metrics_pre_rerank,
        "retrieval_metrics_post_rerank": retrieval_metrics_post_rerank,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "citation_completeness": citation_completeness,
        "unsupported_claim_rate": safety["unsupported_claim_rate"],
        "hallucination_rate": safety["hallucination_rate"],
        "cited_source_ids": list(cited_ids),
        "expected_source_ids": list(expected_ids),
        "fabricated_count": verification.get("fabricated_count", 0),
        "numeric_mismatch_count": verification.get("numeric_mismatch_count", 0),
        "total_citations": verification.get("total_citations", 0),
        "regenerated": result["regenerated"],
        "answer_text": result["answer_text"],
    }


def average_metric(results: list[dict], key: str, nested_key: str = None) -> float:
    """Average a metric across questions where it was actually computed (skip Nones)."""
    values = []
    for r in results:
        val = r.get(key)
        if nested_key and val is not None:
            val = val.get(nested_key)
        if val is not None:
            values.append(val)
    return round(sum(values) / len(values), 3) if values else None


def compute_aggregate_metrics(results: list[dict]) -> dict:
    total = len(results)

    answerable_results = [r for r in results if r["expected_answerable"]]
    unanswerable_results = [r for r in results if not r["expected_answerable"]]

    system_refused = [r for r in results if r["actual_refused"]]
    refusal_precision = (
        sum(1 for r in system_refused if not r["expected_answerable"]) / len(system_refused) if system_refused else None
    )

    should_refuse = [r for r in results if not r["expected_answerable"]]
    refusal_recall = (
        sum(1 for r in should_refuse if r["actual_refused"]) / len(should_refuse) if should_refuse else None
    )

    total_citations_all = sum(r["total_citations"] for r in results)
    total_fabricated = sum(r["fabricated_count"] for r in results)
    total_numeric_mismatch = sum(r["numeric_mismatch_count"] for r in results)

    fabrication_rate = total_fabricated / total_citations_all if total_citations_all else 0
    overall_accuracy = sum(1 for r in results if r["refusal_correct"]) / total if total else 0

    # PER-QUESTION hallucination/unsupported rates, in addition to the
    # per-citation ones already computed via average_metric() above.
    #
    # WHY BOTH: per-citation rates (hallucination_rate, unsupported_claim_rate
    # from average_metric) can be dominated by a single verbose answer — e.g.
    # one multi-document synthesis question that generates 19 citations in one
    # long paragraph counts 19x as much as a question with 1 citation, even
    # though it's still just ONE question out of the whole eval set. This
    # matters in practice: we observed exactly this with eval_017, whose
    # citation count swung the pooled hallucination_rate significantly
    # between runs just because Gemini's non-deterministic output happened to
    # be longer or shorter that time. Per-question rates ("what fraction of
    # QUESTIONS had at least one hallucination") give every question equal
    # weight regardless of how many citations it happened to produce — a
    # fairer summary of how often the system fails, not how many individual
    # citations pile up when it does fail.
    questions_with_any_hallucination = sum(
        1 for r in results if (r["fabricated_count"] + r["numeric_mismatch_count"]) > 0
    )
    per_question_hallucination_rate = questions_with_any_hallucination / total if total else 0

    return {
        "total_questions": total,
        "answerable_questions": len(answerable_results),
        "unanswerable_questions": len(unanswerable_results),
        "--- SAFETY (refusal behavior) ---": "---",
        "overall_refusal_decision_accuracy": round(overall_accuracy, 3),
        "refusal_precision": round(refusal_precision, 3) if refusal_precision is not None else None,
        "refusal_recall": round(refusal_recall, 3) if refusal_recall is not None else None,
        "--- RETRIEVAL (pre-rerank, hybrid fusion order) ---": "---",
        "recall_at_5_pre_rerank": average_metric(answerable_results, "retrieval_metrics_pre_rerank", "recall_at_k"),
        "precision_at_5_pre_rerank": average_metric(
            answerable_results, "retrieval_metrics_pre_rerank", "precision_at_k"
        ),
        "mrr_pre_rerank": average_metric(answerable_results, "retrieval_metrics_pre_rerank", "mrr"),
        "ndcg_at_5_pre_rerank": average_metric(answerable_results, "retrieval_metrics_pre_rerank", "ndcg_at_k"),
        "--- RETRIEVAL (post-rerank, final evidence) ---": "---",
        "recall_at_5_post_rerank": average_metric(answerable_results, "retrieval_metrics_post_rerank", "recall_at_k"),
        "precision_at_5_post_rerank": average_metric(
            answerable_results, "retrieval_metrics_post_rerank", "precision_at_k"
        ),
        "mrr_post_rerank": average_metric(answerable_results, "retrieval_metrics_post_rerank", "mrr"),
        "ndcg_at_5_post_rerank": average_metric(answerable_results, "retrieval_metrics_post_rerank", "ndcg_at_k"),
        "--- GENERATION QUALITY ---": "---",
        "faithfulness": average_metric(results, "faithfulness"),
        "answer_relevance": average_metric(results, "answer_relevance"),
        "citation_completeness": average_metric(results, "citation_completeness"),
        "--- SAFETY (citation-level, per-citation pooled — can be skewed by one verbose answer) ---": "---",
        "unsupported_claim_rate": average_metric(results, "unsupported_claim_rate"),
        "hallucination_rate_per_citation": average_metric(results, "hallucination_rate"),
        "fabrication_rate": round(fabrication_rate, 4),
        "numeric_mismatch_rate": round(total_numeric_mismatch / total_citations_all, 4) if total_citations_all else 0,
        "--- SAFETY (per-question, fairer summary) ---": "---",
        "hallucination_rate_per_question": round(per_question_hallucination_rate, 3),
        "questions_that_triggered_regeneration": sum(1 for r in results if r["regenerated"]),
    }


def print_report(results: list[dict], metrics: dict):
    print("\n" + "=" * 70)
    print("PER-QUESTION RESULTS")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["refusal_correct"] else "FAIL"
        print(
            f"[{status}] [{r['id']}] ({r['question_type']}) refused={r['actual_refused']} "
            f"expected_answerable={r['expected_answerable']}"
        )
        if r["fabricated_count"] > 0 or r["numeric_mismatch_count"] > 0:
            print(f"    WARNING: fabricated={r['fabricated_count']} numeric_mismatch={r['numeric_mismatch_count']}")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    for key, value in metrics.items():
        if value == "---":
            print(f"\n{key}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("WHAT WENT WRONG (for debugging)")
    print("=" * 70)
    wrong = [r for r in results if not r["refusal_correct"]]
    if not wrong:
        print("  Nothing — all refusal decisions were correct.")
    for r in wrong:
        print(f"  [{r['id']}] Expected answerable={r['expected_answerable']}, but system refused={r['actual_refused']}")
        print(f"    Question: {r['question']}")
        print(f"    Answer given: {r['answer_text'][:200]}...")
        print()


def load_checkpoint() -> list[dict]:
    """
    Load any previously completed results, so a run interrupted by (e.g.) a
    daily quota wall can resume from where it left off instead of
    re-spending API calls on questions already answered.
    """
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list[dict]):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run():
    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} evaluation questions")

    results = load_checkpoint()
    completed_ids = {r["id"] for r in results}
    remaining = [item for item in eval_set if item["id"] not in completed_ids]

    if completed_ids:
        print(f"Resuming from checkpoint: {len(completed_ids)} already completed, {len(remaining)} remaining")

    if not remaining:
        print("All questions already completed in checkpoint. Delete eval_checkpoint.json to start fresh.")
    else:
        print("Loading embedding model (used for the answer_relevance metric)...")
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODELS, ACTIVE_EMBEDDING_MODEL

        embedding_model = SentenceTransformer(EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL])

        print("Running each through the live pipeline — this will take a while (real API calls)...\n")

        for i, eval_item in enumerate(remaining, start=1):
            print(f"[{i}/{len(remaining)}] Running: {eval_item['id']} ({eval_item['question_type']})...")
            try:
                result = evaluate_single_question(eval_item, embedding_model)
                results.append(result)
                save_checkpoint(results)  # save after EVERY question, not just at the end

                # Proactive pacing: a small pause between questions to stay
                # under the free-tier per-minute limit BEFORE hitting it,
                # rather than only reacting after a 429 (which is slower
                # overall once you account for retry wait times).
                time.sleep(4)
            except RuntimeError as e:
                error_str = str(e)
                if "GEMINI_DAILY_QUOTA_EXHAUSTED" in error_str or "GEMINI_RATE_LIMIT_PERSISTENT" in error_str:
                    reason = "Daily API quota exhausted" if "DAILY" in error_str else "Persistent rate limiting"
                    print(f"\n{'=' * 70}")
                    print(f"STOPPING: {reason}.")
                    print(f"Progress saved: {len(results)}/{len(eval_set)} questions completed.")
                    if "DAILY" in error_str:
                        print("Run 'python -m src.run_evaluation' again tomorrow (or with a fresh")
                        print("API key) to continue from where this left off.")
                    else:
                        print("Wait a few minutes, then run 'python -m src.run_evaluation' again")
                        print("to continue from where this left off (checkpoint is saved).")
                    print(f"{'=' * 70}")
                    return
                raise
            except Exception as e:
                print(f"  ERROR on {eval_item['id']}: {e}")
                results.append(
                    {
                        "id": eval_item["id"],
                        "question_type": eval_item["question_type"],
                        "question": eval_item["question"],
                        "expected_answerable": eval_item["expected_answerable"],
                        "actual_refused": None,
                        "refusal_correct": False,
                        "retrieval_metrics_pre_rerank": None,
                        "retrieval_metrics_post_rerank": None,
                        "faithfulness": None,
                        "answer_relevance": None,
                        "citation_completeness": None,
                        "unsupported_claim_rate": None,
                        "hallucination_rate": None,
                        "cited_source_ids": [],
                        "expected_source_ids": eval_item["expected_source_ids"],
                        "fabricated_count": 0,
                        "numeric_mismatch_count": 0,
                        "total_citations": 0,
                        "regenerated": False,
                        "answer_text": f"ERROR: {e}",
                    }
                )
                save_checkpoint(results)

    metrics = compute_aggregate_metrics(results)
    print_report(results, metrics)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)
    print(f"\nFull results saved to: {RESULTS_PATH}")

    # --- Phase 16: log this run as an experiment ---
    log_experiment(metrics, notes=f"{len(eval_set)}-question eval set")
    print_experiment_table()


if __name__ == "__main__":
    run()
