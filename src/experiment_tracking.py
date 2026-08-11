"""
experiment_tracking.py

Phase 16: Experiment Tracking.

DECISION: a lightweight CSV log instead of MLflow.

MLflow (or similar tools like Weights & Biases) is built for teams running
many parallel experiments, comparing model variants at scale, and often
serving models afterward — it requires its own tracking server, a storage
backend, and a UI to be useful. For a solo portfolio project running
occasional evaluation passes, that's meaningfully more infrastructure than
the task needs. A single CSV file satisfies every actual requirement here:
it records every experiment's configuration and results, and it's trivial
to load into a spreadsheet or pandas DataFrame for comparison — which is
the same thing MLflow's UI would show for a project this size. If this
project later needed to compare dozens of embedding models or chunking
strategies in parallel, or needed a team-shared dashboard, that would be
the point to introduce MLflow — not before.

Each row in the log captures a full "experiment configuration snapshot"
(embedding model, chunking method, retrieval method, top_k, reranker, LLM,
prompt version) alongside the evaluation results produced by that exact
configuration, so you can always answer "what settings produced these
numbers" for any past run.
"""

import csv
from datetime import datetime, timezone

from src.config import (
    PROJECT_ROOT,
    ACTIVE_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    GEMINI_MODEL_NAME,
)
from src.rag_pipeline import PROMPT_VERSION

EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_LOG_PATH = EXPERIMENTS_DIR / "experiments.csv"

CHUNKING_METHOD = "section-aware (structural split on Inclusion/Exclusion for eligibility criteria, size-based fallback ~180 words/chunk with 30-word overlap for other long sections, pass-through for short sections)"
CHUNK_SIZE_MAX_WORDS = 180
RETRIEVAL_METHOD = "hybrid: semantic (embeddings) + BM25 keyword, fused via Reciprocal Rank Fusion, plus explicit NCT/PMID ID injection when named directly in the question"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2, with pinning for explicitly-named source IDs"

FIELDNAMES = [
    "timestamp",
    "experiment_id",
    "embedding_model",
    "chunking_method",
    "chunk_size_max_words",
    "retrieval_method",
    "top_k",
    "candidate_pool_size",
    "reranker_model",
    "llm_model",
    "prompt_version",
    "overall_refusal_decision_accuracy",
    "refusal_precision",
    "refusal_recall",
    "faithfulness",
    "answer_relevance",
    "citation_completeness",
    "fabrication_rate",
    "hallucination_rate_per_question",
    "recall_at_5_post_rerank",
    "mrr_post_rerank",
    "notes",
]


def build_experiment_record(metrics: dict, top_k: int = 5, candidate_pool_size: int = 15, notes: str = "") -> dict:
    """Assemble one experiment record: config snapshot + the metrics it produced."""
    experiment_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "embedding_model": EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL],
        "chunking_method": CHUNKING_METHOD,
        "chunk_size_max_words": CHUNK_SIZE_MAX_WORDS,
        "retrieval_method": RETRIEVAL_METHOD,
        "top_k": top_k,
        "candidate_pool_size": candidate_pool_size,
        "reranker_model": RERANKER_MODEL,
        "llm_model": GEMINI_MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "overall_refusal_decision_accuracy": metrics.get("overall_refusal_decision_accuracy"),
        "refusal_precision": metrics.get("refusal_precision"),
        "refusal_recall": metrics.get("refusal_recall"),
        "faithfulness": metrics.get("faithfulness"),
        "answer_relevance": metrics.get("answer_relevance"),
        "citation_completeness": metrics.get("citation_completeness"),
        "fabrication_rate": metrics.get("fabrication_rate"),
        "hallucination_rate_per_question": metrics.get("hallucination_rate_per_question"),
        "recall_at_5_post_rerank": metrics.get("recall_at_5_post_rerank"),
        "mrr_post_rerank": metrics.get("mrr_post_rerank"),
        "notes": notes,
    }


def log_experiment(metrics: dict, top_k: int = 5, candidate_pool_size: int = 15, notes: str = ""):
    """Append one experiment record to the CSV log."""
    record = build_experiment_record(metrics, top_k, candidate_pool_size, notes)

    file_exists = EXPERIMENTS_LOG_PATH.exists()
    with open(EXPERIMENTS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    print(f"\nExperiment logged: {record['experiment_id']}")
    return record


def print_experiment_table():
    """Print all past experiments as a simple, readable table."""
    if not EXPERIMENTS_LOG_PATH.exists():
        print("No experiments logged yet.")
        return

    with open(EXPERIMENTS_LOG_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No experiments logged yet.")
        return

    display_columns = [
        "experiment_id",
        "embedding_model",
        "llm_model",
        "prompt_version",
        "overall_refusal_decision_accuracy",
        "refusal_recall",
        "faithfulness",
        "fabrication_rate",
        "hallucination_rate_per_question",
    ]

    print("\n" + "=" * 100)
    print("EXPERIMENT HISTORY")
    print("=" * 100)

    header = " | ".join(f"{col[:18]:18}" for col in display_columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = " | ".join(f"{str(row.get(col, ''))[:18]:18}" for col in display_columns)
        print(line)

    print(f"\nFull details (all {len(FIELDNAMES)} fields) saved in: {EXPERIMENTS_LOG_PATH}")


if __name__ == "__main__":
    print_experiment_table()
