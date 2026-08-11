"""
build_vector_db.py

Phase 7: Vector Database.

Takes the chunks (Phase 5) and their embeddings (Phase 6) and loads them into
Chroma — a local, file-based vector database. Once loaded, we can hand it a
brand-new question, and it will instantly find the most similar chunks out
of all 4,161, without us writing any manual similarity-comparison code
ourselves (unlike the quick "sanity check" we hand-rolled in Phase 6).

Why Chroma for this project: no separate server process to run (unlike
Qdrant or Postgres+pgvector), works as a plain Python library, and stores
everything in a local folder — the right level of complexity for a
4,000-chunk portfolio project. A production system with millions of
documents would likely outgrow this and move to something like Qdrant or
pgvector, but that's a "later, if needed" decision, not a "now" one.

Usage:
    python -m src.build_vector_db          # builds the database
    python -m src.build_vector_db --query "what are common exclusion criteria for hypertension trials"
"""

import argparse
import json

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, EMBEDDING_MODELS, ACTIVE_EMBEDDING_MODEL

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
COLLECTION_NAME = "clinical_rag_chunks"


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sanitize_metadata(chunk: dict) -> dict:
    """
    Chroma only accepts metadata values that are str, int, float, or bool —
    no lists, dicts, or None. Our chunks have nested metadata (e.g. "phases"
    is a list, "mesh_terms" is a list, some fields can be None), so we flatten
    everything into simple types here before handing it to Chroma.
    """
    flat = {
        "document_id": chunk["document_id"],
        "source_type": chunk["source_type"],
        "source_id": chunk["source_id"] or "",
        "title": chunk["title"] or "",
        "condition_query": chunk["condition_query"],
        "section_name": chunk["section_name"],
        "word_count": chunk["word_count"],
    }

    original_metadata = chunk.get("metadata") or {}
    for key, value in original_metadata.items():
        if value is None:
            flat[key] = ""
        elif isinstance(value, list):
            flat[key] = ", ".join(str(v) for v in value)  # e.g. ["PHASE2","PHASE3"] -> "PHASE2, PHASE3"
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)

    return flat


def build_database():
    print(f"Active embedding model: {ACTIVE_EMBEDDING_MODEL} ({EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]})")

    vectors_path = EMBEDDINGS_DIR / f"embeddings_{ACTIVE_EMBEDDING_MODEL}.npy"
    ids_path = EMBEDDINGS_DIR / f"chunk_ids_{ACTIVE_EMBEDDING_MODEL}.json"

    if not vectors_path.exists():
        print(
            f"ERROR: {vectors_path} not found. Run 'python -m src.embed_corpus' first "
            f"with ACTIVE_EMBEDDING_MODEL='{ACTIVE_EMBEDDING_MODEL}' in config.py."
        )
        return

    embeddings = np.load(vectors_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        embedded_chunk_ids = json.load(f)

    chunks = load_jsonl(CHUNKS_PATH)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    print(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings")

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Start fresh each time this script runs, so re-running it after data
    # changes doesn't leave stale/duplicate entries behind.
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"Removed existing '{COLLECTION_NAME}' collection to rebuild fresh")

    collection = client.create_collection(name=COLLECTION_NAME)

    # Insert in batches — large single inserts can be slow/fragile.
    batch_size = 500
    for start in range(0, len(embedded_chunk_ids), batch_size):
        end = start + batch_size
        batch_ids = embedded_chunk_ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()
        batch_documents = [chunks_by_id[cid]["text"] for cid in batch_ids]
        batch_metadatas = [sanitize_metadata(chunks_by_id[cid]) for cid in batch_ids]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print(
            f"  Inserted {end if end < len(embedded_chunk_ids) else len(embedded_chunk_ids)}/{len(embedded_chunk_ids)} chunks"
        )

    print(f"\nVector database built at: {VECTOR_DB_DIR}")
    print(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} chunks")


def query_database(question: str, top_k: int = 5, condition_filter: str = None):
    """
    Embed a question with the SAME model used to build the database (critical —
    mixing embeddings from two different models produces meaningless results,
    since each model has its own "meaning space" that isn't compatible with
    another model's), then retrieve the top_k most similar chunks.
    """
    model_name = EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]
    print(f"Embedding query with: {model_name}")
    model = SentenceTransformer(model_name)
    query_vector = model.encode([question], convert_to_numpy=True)[0].tolist()

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    where_clause = {"condition_query": condition_filter} if condition_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_clause,
    )

    print(f'\nQuery: "{question}"')
    if condition_filter:
        print(f"Filtered to condition: {condition_filter}")
    print(f"\nTop {top_k} results:\n")

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (chunk_id, text, meta, distance) in enumerate(zip(ids, documents, metadatas, distances), start=1):
        similarity = 1 - distance  # Chroma returns distance; smaller = more similar
        print(f"[{rank}] {chunk_id}  (similarity: {similarity:.3f})")
        print(f"    Source: {meta['source_type']} | {meta['title'][:80]}")
        print(f"    Section: {meta['section_name']}")
        print(f"    Text: {text[:200]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None, help="Run a test query instead of rebuilding the database")
    parser.add_argument("--condition", type=str, default=None, help="Optional: filter query results to one condition")
    args = parser.parse_args()

    if args.query:
        query_database(args.query, condition_filter=args.condition)
    else:
        build_database()
