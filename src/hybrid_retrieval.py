"""
hybrid_retrieval.py

Phase 8: Hybrid Retrieval.

Combines two different search methods, because each catches things the
other misses:

  - SEMANTIC search (Phase 7's Chroma database): finds chunks with similar
    MEANING, even if the exact words differ. Good for conceptual questions
    like "what side effects were reported" matching text that says
    "adverse events" instead.

  - KEYWORD search (BM25, a classic exact-term-matching algorithm): finds
    chunks containing the EXACT words in the query. Critical for things
    semantic search can be surprisingly bad at: specific drug names, NCT
    IDs, exact numbers, rare medical terms that weren't well-represented in
    the embedding model's training data.

The two ranked lists are combined using Reciprocal Rank Fusion (RRF) — a
simple, well-established method: a chunk's combined score is the sum of
1/(60 + rank) across both methods. Chunks that rank well in BOTH methods
naturally rise to the top; chunks that only one method liked still get a
fair chance, rather than being drowned out by raw score-scale differences
between semantic similarity and BM25 scores (which aren't directly
comparable numbers).

We also add a confidence check based on what Phase 7 testing revealed: when
even the TOP semantic result has a low similarity score, that's a signal the
corpus may not have great evidence for this question. We surface that
honestly instead of presenting weak results as if they were strong ones.

Usage:
    python -m src.hybrid_retrieval --query "what were the exclusion criteria for hypertension trials"
    python -m src.hybrid_retrieval --query "cardiovascular outcomes" --condition "type 2 diabetes"
"""

import argparse
import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, EMBEDDING_MODELS, ACTIVE_EMBEDDING_MODEL

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
COLLECTION_NAME = "clinical_rag_chunks"

# Below this semantic similarity score, we warn that evidence may be weak.
# Chosen based on Phase 7 testing: our strong match was ~0.5, and results
# below ~0.15-0.2 were clearly tangential rather than genuinely relevant.
LOW_CONFIDENCE_THRESHOLD = 0.30

RRF_K = 60  # standard constant used in Reciprocal Rank Fusion


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer for BM25 — good enough for this scale."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks: list[dict]):
    """Build a BM25 index over all chunk texts, for exact keyword matching."""
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def semantic_search(query: str, model, collection, condition_filter: str, top_k: int):
    query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()
    where_clause = {"condition_query": condition_filter} if condition_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_clause,
    )

    ranked = []
    for rank, (chunk_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0]), start=1):
        similarity = 1 - distance
        ranked.append({"chunk_id": chunk_id, "rank": rank, "score": similarity})
    return ranked


def keyword_search(query: str, chunks: list[dict], bm25_index, condition_filter: str, top_k: int):
    query_tokens = tokenize(query)
    all_scores = bm25_index.get_scores(query_tokens)

    # Pair each chunk with its score, optionally filtering by condition first
    scored = [
        (chunks[i]["chunk_id"], all_scores[i])
        for i in range(len(chunks))
        if not condition_filter or chunks[i]["condition_query"] == condition_filter
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for rank, (chunk_id, score) in enumerate(scored[:top_k], start=1):
        ranked.append({"chunk_id": chunk_id, "rank": rank, "score": float(score)})
    return ranked


def reciprocal_rank_fusion(semantic_ranked: list[dict], keyword_ranked: list[dict]) -> dict:
    """
    Combine two ranked lists into one fused score per chunk_id.
    A chunk ranked #1 in both lists scores much higher than one ranked #1
    in only one list — this is what makes RRF a genuine "hybrid" rather
    than just picking whichever method found more results.
    """
    fused_scores = {}

    for entry in semantic_ranked:
        fused_scores[entry["chunk_id"]] = fused_scores.get(entry["chunk_id"], 0) + 1 / (RRF_K + entry["rank"])

    for entry in keyword_ranked:
        fused_scores[entry["chunk_id"]] = fused_scores.get(entry["chunk_id"], 0) + 1 / (RRF_K + entry["rank"])

    return fused_scores


def hybrid_search_raw(
    question: str, condition_filter: str = None, top_k: int = 5, candidate_pool: int = 20, return_candidates: int = None
) -> dict:
    """
    Same hybrid retrieval logic as hybrid_search(), but returns structured
    data instead of printing it — this is what rag_pipeline.py (Phase 9)
    calls to get evidence chunks it can feed into the LLM prompt.

    return_candidates: if set, return this many chunks (for a downstream
    reranker to further narrow down) instead of cutting straight to top_k.
    This lets the pipeline say "give me the top 15 hybrid-retrieval results,
    THEN I'll rerank those down to the final top 5" rather than committing
    to the final 5 before reranking even runs.
    """
    model_name = EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    chunks = load_jsonl(CHUNKS_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    bm25_index = build_bm25_index(chunks)

    semantic_ranked = semantic_search(question, model, collection, condition_filter, candidate_pool)
    keyword_ranked = keyword_search(question, chunks, bm25_index, condition_filter, candidate_pool)

    fused_scores = reciprocal_rank_fusion(semantic_ranked, keyword_ranked)
    cutoff = return_candidates if return_candidates else top_k
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:cutoff]

    semantic_scores_by_id = {e["chunk_id"]: e["score"] for e in semantic_ranked}
    top_semantic_score = max((semantic_scores_by_id.get(cid, 0) for cid in sorted_chunk_ids), default=0)

    result_chunks = [chunks_by_id[cid] for cid in sorted_chunk_ids]

    return {
        "chunks": result_chunks,
        "top_semantic_score": top_semantic_score,
        "low_confidence": top_semantic_score < LOW_CONFIDENCE_THRESHOLD,
    }


def hybrid_search(question: str, condition_filter: str = None, top_k: int = 5, candidate_pool: int = 20):
    model_name = EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    chunks = load_jsonl(CHUNKS_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    bm25_index = build_bm25_index(chunks)

    semantic_ranked = semantic_search(question, model, collection, condition_filter, candidate_pool)
    keyword_ranked = keyword_search(question, chunks, bm25_index, condition_filter, candidate_pool)

    fused_scores = reciprocal_rank_fusion(semantic_ranked, keyword_ranked)
    sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    # Confidence check: look at the BEST raw semantic similarity score among
    # our final results (not the fused score, which isn't on a 0-1 scale).
    semantic_scores_by_id = {e["chunk_id"]: e["score"] for e in semantic_ranked}
    top_semantic_score = max((semantic_scores_by_id.get(cid, 0) for cid in sorted_chunk_ids), default=0)

    print(f'\nQuery: "{question}"')
    if condition_filter:
        print(f"Filtered to condition: {condition_filter}")

    if top_semantic_score < LOW_CONFIDENCE_THRESHOLD:
        print(
            f"\n⚠️  LOW CONFIDENCE WARNING: best semantic match score is only "
            f"{top_semantic_score:.3f} (threshold: {LOW_CONFIDENCE_THRESHOLD})."
        )
        print("   The retrieved evidence below may not strongly support a confident answer.")
        print("   A production system should refuse or hedge here rather than answer plainly (see Phase 11).")

    print(f"\nTop {len(sorted_chunk_ids)} hybrid results:\n")
    for rank, chunk_id in enumerate(sorted_chunk_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        sem_score = semantic_scores_by_id.get(chunk_id)
        sem_display = f"{sem_score:.3f}" if sem_score is not None else "not in semantic top-k"

        print(f"[{rank}] {chunk_id}")
        print(f"    Fused score: {fused_scores[chunk_id]:.4f}  |  Semantic score: {sem_display}")
        print(f"    Source: {chunk['source_type']} | {chunk['title'][:80]}")
        print(f"    Section: {chunk['section_name']}")
        print(f"    Text: {chunk['text'][:200]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--condition", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    hybrid_search(args.query, condition_filter=args.condition, top_k=args.top_k)
