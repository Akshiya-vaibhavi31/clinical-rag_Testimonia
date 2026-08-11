"""
reranker.py

Fills Gap 1 from the Phase 9 diagram: a real RERANKING stage.

Our hybrid retrieval (Phase 8) uses embeddings + BM25 to quickly narrow
thousands of chunks down to a candidate pool — this is fast, but embeddings
compare a question and a chunk INDEPENDENTLY (each gets its own vector,
computed without seeing the other). That's efficient but less precise.

A cross-encoder reranker instead looks at the question and ONE candidate
chunk TOGETHER, in a single pass, and outputs a direct relevance score. This
is much slower (must run once per candidate — can't be pre-computed and
cached like embeddings can), which is exactly why we don't use it on all
4,161 chunks. Instead: hybrid retrieval finds ~15-20 reasonable candidates
fast, then the reranker carefully re-scores just those few, more accurately.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2 — a small, fast, widely
used reranker trained specifically for "is this passage relevant to this
query" style ranking (from the MS MARCO passage ranking dataset). It's
general-purpose, not biomedical-specific — a reasonable, honest limitation
to note: a biomedical cross-encoder would likely do better, but none as
well-established and easy to run locally exists at this model's size.
"""

from sentence_transformers import CrossEncoder

_reranker_model = None  # loaded once, reused across calls (loading is slow)


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        print("Loading reranker model (first call only)...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank_chunks(question: str, chunks: list[dict], top_k: int = 5, pinned_source_ids: list[str] = None) -> list[dict]:
    """
    Re-score a list of candidate chunks against the question using the
    cross-encoder, and return the top_k best, sorted by the new score.

    Each chunk dict gets a new "rerank_score" field added, so we can inspect
    and report on it later (e.g. in evaluation, or just for transparency in
    the pipeline output).

    pinned_source_ids: source IDs that MUST appear in the final result,
    regardless of their cross-encoder score.

    WHY THIS MATTERS (found via real evaluation results): when a question
    explicitly names a trial by ID, we guarantee its chunks reach the
    CANDIDATE pool (see detect_explicit_ids in rag_pipeline.py) — but the
    reranker itself has no concept of "this is the exact trial the user
    asked about." It just scores every candidate on semantic similarity to
    the question text, which can rank a DIFFERENT trial with superficially
    similar wording higher than the actual trial being asked about. This
    caused a real failure: asking about NCT01595789 by name still resulted
    in three OTHER liraglutide trials filling the top 5, while NCT01595789's
    own chunks scored lower and got dropped. Pinning guarantees an exact
    match survives to the final evidence set regardless of its score.
    """
    if not chunks:
        return []

    model = get_reranker()

    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    if not pinned_source_ids:
        return reranked[:top_k]

    pinned_ids_upper = [pid.upper() for pid in pinned_source_ids]
    pinned_chunks = [c for c in reranked if c["source_id"].upper() in pinned_ids_upper]
    other_chunks = [c for c in reranked if c["source_id"].upper() not in pinned_ids_upper]

    # Cap how many chunks we pin PER explicitly-named source, rather than
    # including every chunk that source has unconditionally.
    #
    # WHY (found via real eval results): pinning ALL of a trial's chunks
    # (sometimes 7+ for a trial with many eligibility criteria sections)
    # ballooned evidence set size well past top_k, which correlated with
    # WORSE generation safety metrics in testing — faithfulness dropped and
    # regenerations roughly doubled. More citations from densely-overlapping
    # sections of the same trial gives our numeric-verification heuristic
    # more surface area to produce false positives (see
    # citation_verification.py's documented limitations). Since pinned
    # chunks are already sorted by rerank_score (best-matching sections of
    # that trial first), capping to the best few per source preserves the
    # retrieval guarantee (the named trial WILL appear) while controlling
    # evidence bloat.
    MAX_CHUNKS_PER_PINNED_SOURCE = 3
    capped_pinned_chunks = []
    per_source_count = {}
    for chunk in pinned_chunks:
        sid = chunk["source_id"].upper()
        if per_source_count.get(sid, 0) < MAX_CHUNKS_PER_PINNED_SOURCE:
            capped_pinned_chunks.append(chunk)
            per_source_count[sid] = per_source_count.get(sid, 0) + 1

    # Pinned chunks go first (guaranteed inclusion), then fill remaining
    # slots with the best-scoring other chunks, capped at top_k total.
    remaining_slots = max(0, top_k - len(capped_pinned_chunks))
    return capped_pinned_chunks + other_chunks[:remaining_slots]
