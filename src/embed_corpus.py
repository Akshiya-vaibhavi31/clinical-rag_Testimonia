"""
embed_corpus.py

Phase 6: Generate embeddings for every chunk.

Reads data/processed/chunks.jsonl (built in Phase 5) and converts each
chunk's text into an embedding vector — a list of numbers that captures its
meaning, which is what makes semantic search possible later (Phase 7-8).

We use the `sentence-transformers` library, which handles downloading and
running the embedding model locally (no API key or internet cost per call —
only the one-time model download).

Try this with BOTH models in config.py (ACTIVE_EMBEDDING_MODEL = "general"
then "biomedical") to compare speed and file size, as part of the Phase 6
model selection process. Actual retrieval QUALITY comparison happens in
Phase 15, once we have an evaluation question set.

Usage:
    python -m src.embed_corpus
"""

import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import PROJECT_ROOT, EMBEDDING_MODELS, ACTIVE_EMBEDDING_MODEL

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# Output files are named after which model produced them, so embeddings from
# different models never accidentally get mixed up or overwritten.
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run():
    model_name = EMBEDDING_MODELS[ACTIVE_EMBEDDING_MODEL]
    print(f"Using model: {model_name} (key: '{ACTIVE_EMBEDDING_MODEL}')")
    print("Loading model (first run downloads it — may take a minute)...")

    load_start = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f} seconds")

    chunks = load_jsonl(CHUNKS_PATH)
    print(f"\nLoaded {len(chunks)} chunks to embed")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    print("Generating embeddings (this is the slow part, be patient)...")
    embed_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embed_time = time.time() - embed_start

    print(f"\nEmbedding generation took {embed_time:.1f} seconds")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Total vectors created: {embeddings.shape[0]}")
    print(f"Average time per chunk: {(embed_time / len(chunks)) * 1000:.1f} ms")

    # Save the embeddings matrix and the chunk_id list that maps each row
    # back to its source chunk. We save chunk_ids separately (not baked into
    # the .npy file) because numpy arrays only hold numbers, not text.
    vectors_path = EMBEDDINGS_DIR / f"embeddings_{ACTIVE_EMBEDDING_MODEL}.npy"
    ids_path = EMBEDDINGS_DIR / f"chunk_ids_{ACTIVE_EMBEDDING_MODEL}.json"

    np.save(vectors_path, embeddings)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    print(f"\nEmbeddings saved to: {vectors_path}")
    print(f"Chunk ID mapping saved to: {ids_path}")

    # Quick sanity check: embed one query and show which existing chunk is
    # most similar, using basic cosine similarity computed by hand. This
    # doesn't replace Phase 7's real vector database, but it confirms the
    # embeddings are actually meaningful before we build more on top of them.
    print("\n--- Sanity check ---")
    sample_query = "eligibility criteria for diabetes trial"
    query_vector = model.encode([sample_query], convert_to_numpy=True)[0]

    # Cosine similarity: how "close in meaning" two vectors are (1.0 = identical direction)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vector)
    similarities = (embeddings @ query_vector) / norms
    best_idx = int(np.argmax(similarities))

    print(f'Query: "{sample_query}"')
    print(f"Most similar chunk (score {similarities[best_idx]:.3f}): {chunk_ids[best_idx]}")
    print(f"Text: {texts[best_idx][:200]}")


if __name__ == "__main__":
    run()
