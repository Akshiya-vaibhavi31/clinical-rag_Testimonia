"""
evaluation_metrics.py

Phase 15: the full metric suite, mapped explicitly to the categories the
project brief requires and to their RAGAS-equivalent names where applicable.

WHY NOT JUST USE RAGAS DIRECTLY (investigated, as the brief asked):
RAGAS (Es et al. 2023) is the canonical open-source RAG evaluation library,
built around four core metrics: faithfulness, answer relevance, context
precision, and context recall. It computes these using LLM-AS-JUDGE — i.e.
it makes its own separate API calls to an LLM to score each metric. For our
32-question eval set, that would roughly double our Gemini API usage per
run (one call to answer the question, another to judge it) for a project
already on a free tier with sensible daily limits. We chose to implement
lightweight, LOCAL heuristic equivalents instead — cheaper and faster, but
genuinely less accurate than a real LLM-judge. This tradeoff is disclosed
per-metric below. A production system with budget would likely use real
RAGAS or a similar LLM-judge framework instead of these heuristics.

RETRIEVAL METRICS (all use binary relevance: a chunk is "relevant" only if
its source_id is in the eval question's expected_source_ids — a real
limitation, since we don't have full relevance judgments for every chunk,
only whether the ONE expected source was found):

  - Recall@k: was at least one expected source found in the top k results?
  - Precision@k: what fraction of the top k results were the expected source?
    (weak with only 1-2 relevant docs per question — most "irrelevant" chunks
    in the top k might actually be reasonably relevant, we just don't have
    labels for them, so this systematically underestimates precision)
  - MRR (Mean Reciprocal Rank): 1/rank of the first expected source found
    (0 if not found at all) — rewards ranking the right answer FIRST, not
    just somewhere in the top k
  - NDCG (Normalized Discounted Cumulative Gain): similar to MRR but
    accounts for multiple relevant documents and gives partial credit for
    lower positions — with only binary relevance and often 1 relevant doc
    per question, NDCG here is closely related to MRR, not meaningfully
    different; a real NDCG needs graded relevance judgments to show its
    value over simpler metrics.

GENERATION METRICS:

  - Faithfulness: fraction of citations that are genuinely verified (real
    source, good lexical/numeric support). RAGAS's faithfulness works by
    breaking the answer into individual claims and checking each against
    context using an LLM judge — ours checks at the CITATION level using
    lexical/numeric heuristics, a coarser but much cheaper proxy.
  - Answer relevance: embedding cosine similarity between the question and
    the generated answer. RAGAS's version generates synthetic questions
    FROM the answer and compares them to the original — a more rigorous
    approach we're not implementing here. Simple cosine similarity has a
    known weakness: a well-worded REFUSAL can score artificially low on
    this metric even though refusing was the objectively correct behavior,
    since a refusal's wording doesn't lexically/semantically resemble the
    question the way a direct answer would.
  - Citation correctness: of the citations made, what fraction accurately
    represent their source (same computation as faithfulness in our
    implementation — a real system might define these more distinctly).
  - Citation completeness: of the sentences making a factual claim, what
    fraction have a citation attached at all? Heuristic: counts sentences
    containing a number or definitive claim language, checks for a nearby
    citation marker. Will misfire on unusual sentence structures.

SAFETY METRICS:

  - Unsupported claim rate: fraction of citations flagged as "weakly
    supported" (real source, low lexical overlap) — see the documented
    paraphrase-vs-fabrication ambiguity in citation_verification.py.
  - Hallucination rate: fraction of citations that are fabricated OR have
    a numeric mismatch — the two HARD failure types we can detect with
    high confidence.
  - Refusal precision / recall: computed in run_evaluation.py already,
    included here in the printed summary for completeness.
"""

import re
import math


def compute_retrieval_ranking_metrics(ranked_source_ids: list[str], expected_source_ids: set[str], k: int = 5) -> dict:
    """
    Compute Recall@k, Precision@k, MRR, and NDCG@k for one question.

    ranked_source_ids: source IDs in the order they were retrieved/ranked
    (best first). expected_source_ids: the ground-truth relevant sources
    from the eval set.

    BUG FIX (found via real eval results — NDCG scored 1.399, which is
    mathematically impossible since NDCG is always in [0, 1]): these are
    standard IR metrics designed for a ranked list of DISTINCT documents.
    Our reranker's "pinning" feature (which guarantees an explicitly-named
    trial survives reranking) can inject SEVERAL CHUNKS from the SAME
    trial into the evidence set — so the same source_id can appear at
    multiple ranked positions. The original code treated each position as
    an independent relevant "hit," so one trial appearing 5 times in the
    top 5 counted as 5 wins instead of 1, inflating DCG far past what IDCG
    (calculated assuming one relevant item per source) could normalize
    against. Fix: deduplicate the ranked list by source_id BEFORE computing
    any metric, keeping only each source's first (best-ranked) occurrence —
    this correctly evaluates ranking quality at the document level, not the
    chunk level.
    """
    if not expected_source_ids:
        return {"recall_at_k": None, "precision_at_k": None, "mrr": None, "ndcg_at_k": None}

    # Deduplicate by source_id, keeping first (best-ranked) occurrence only
    seen = set()
    deduped_ids = []
    for sid in ranked_source_ids:
        if sid not in seen:
            deduped_ids.append(sid)
            seen.add(sid)
    ranked_source_ids = deduped_ids

    top_k = ranked_source_ids[:k]

    hits_in_k = [sid for sid in top_k if sid in expected_source_ids]
    recall_at_k = 1.0 if hits_in_k else 0.0
    precision_at_k = len(hits_in_k) / len(top_k) if top_k else 0.0

    mrr = 0.0
    for rank, sid in enumerate(ranked_source_ids, start=1):
        if sid in expected_source_ids:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for i, sid in enumerate(top_k, start=1):
        relevance = 1.0 if sid in expected_source_ids else 0.0
        dcg += relevance / math.log2(i + 1)

    num_relevant = min(len(expected_source_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_relevant + 1))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall_at_k": round(recall_at_k, 3),
        "precision_at_k": round(precision_at_k, 3),
        "mrr": round(mrr, 3),
        "ndcg_at_k": round(ndcg_at_k, 3),
    }


def compute_faithfulness(verification_report: dict) -> float:
    """
    Fraction of citations that are genuinely verified (real source, good
    lexical/numeric support). See module docstring for how this differs
    from RAGAS's claim-level LLM-judge approach.
    """
    total = verification_report.get("total_citations", 0)
    if total == 0:
        return None
    verified = verification_report.get("verified_count", 0)
    return round(verified / total, 3)


def compute_answer_relevance(question: str, answer_text: str, embedding_model) -> float:
    """
    Cosine similarity between the question and answer embeddings, as a
    cheap proxy for "does the answer address the question." See module
    docstring for the known refusal-scoring weakness of this approach.
    """
    import numpy as np

    vectors = embedding_model.encode([question, answer_text], convert_to_numpy=True)
    q_vec, a_vec = vectors[0], vectors[1]
    similarity = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec)))
    return round(similarity, 3)


def compute_citation_completeness(answer_text: str) -> float:
    """
    Heuristic: what fraction of sentences that appear to make a factual
    claim (contain a number) have a citation marker "[source: ...]"
    attached within the same sentence?

    This is a rough heuristic, not a reliable claim-extraction system — it
    will both over-count (flagging non-factual sentences as needing
    citations) and under-count (missing claims phrased without numbers).
    """
    citation_pattern = r"\[source:\s*(?:NCT\d+|PMID\s*\d+)\]"
    number_pattern = r"\d"

    sentences = re.split(r"(?<=[.!?])\s+|\n[*\-]\s*", answer_text)
    factual_sentences = [s for s in sentences if re.search(number_pattern, s) and len(s.strip()) > 10]

    if not factual_sentences:
        return None

    cited = sum(1 for s in factual_sentences if re.search(citation_pattern, s))
    return round(cited / len(factual_sentences), 3)


def compute_safety_metrics(verification_report: dict) -> dict:
    """Unsupported claim rate and hallucination rate, from the verification report."""
    total = verification_report.get("total_citations", 0)
    if total == 0:
        return {"unsupported_claim_rate": None, "hallucination_rate": None}

    weakly_supported = verification_report.get("weakly_supported_count", 0)
    fabricated = verification_report.get("fabricated_count", 0)
    numeric_mismatch = verification_report.get("numeric_mismatch_count", 0)

    return {
        "unsupported_claim_rate": round(weakly_supported / total, 3),
        "hallucination_rate": round((fabricated + numeric_mismatch) / total, 3),
    }
