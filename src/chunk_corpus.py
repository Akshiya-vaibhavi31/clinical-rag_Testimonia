"""
chunk_corpus.py

Phase 5: Medical-Aware Chunking.

Reads data/processed/corpus.jsonl (built in Phase 4) and splits each
section's text into "chunks" — smaller pieces of text that a retrieval
system can search over individually, instead of matching against a whole
document (which could be over 1,000 words for eligibility criteria).

Chunking strategy is SECTION-AWARE, based directly on what Phase 3's data
exploration told us:

  - Eligibility criteria: highly variable length (3 to 1,607 words) and
    almost always has a natural internal structure ("Inclusion Criteria:"
    followed by "Exclusion Criteria:"). We split on that boundary FIRST,
    then further split each half by word count if it's still long.

  - Short sections (abstracts, brief summaries): usually well under our
    chunk size limit already (Phase 3 showed abstracts average ~150-260
    words). These are kept as ONE chunk — splitting them further would only
    hurt retrieval by breaking a coherent short passage into fragments.

  - Everything else (detailed descriptions, outcomes, or any section that
    turns out to be long): split by word count with a small overlap between
    consecutive chunks, so a sentence that would otherwise be cut in half
    across a chunk boundary still appears in full in at least one chunk.

Every chunk keeps a reference back to its parent document (document_id,
source_type, source_id, title) so we can always cite the original source
later — this is what makes citation-grounded answers possible in Phase 10+.

Usage:
    python -m src.chunk_corpus
"""

import json
import re

from src.config import PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# A chunk larger than this (in words) gets split further.
# Chosen because Phase 3 showed most abstracts/summaries fall well under
# this, so short sections pass through untouched, while long eligibility
# criteria and detailed descriptions get broken up.
MAX_CHUNK_WORDS = 180
OVERLAP_WORDS = 30  # words repeated between consecutive chunks, to preserve context


def load_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_by_word_count(text: str, max_words: int = MAX_CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """
    Generic fallback splitter: break text into overlapping windows of
    `max_words` words. Used for any section that isn't short enough to
    keep whole, and doesn't have a more specific structure to split on.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step back by `overlap` words so context carries over

    return chunks


def split_eligibility_criteria(text: str) -> list[str]:
    """
    Eligibility criteria almost always follow a predictable structure:
    "Inclusion Criteria:" followed later by "Exclusion Criteria:". Splitting
    on this real structural boundary (rather than blindly by word count)
    keeps each chunk semantically coherent — a question like "what are the
    exclusion criteria?" should retrieve a chunk that IS the exclusion
    criteria, not a chunk that's half inclusion and half exclusion.
    """
    # Case-insensitive search for the boundary between the two sections.
    match = re.search(r"exclusion criteria\s*:?", text, flags=re.IGNORECASE)

    if not match:
        # No clear structure found — fall back to generic word-count splitting.
        return split_by_word_count(text)

    inclusion_part = text[: match.start()].strip()
    exclusion_part = text[match.start() :].strip()

    chunks = []
    if inclusion_part:
        chunks.extend(split_by_word_count(inclusion_part))
    if exclusion_part:
        chunks.extend(split_by_word_count(exclusion_part))

    return chunks if chunks else split_by_word_count(text)


def chunk_section(section_name: str, text: str) -> list[str]:
    """Route each section to the appropriate chunking strategy."""
    if not text or not text.strip():
        return []

    word_count = len(text.split())

    # Short sections stay whole — splitting a 150-word abstract would only
    # produce fragments that individually make less sense than the original.
    if word_count <= MAX_CHUNK_WORDS:
        return [text]

    if section_name == "eligibility_criteria":
        return split_eligibility_criteria(text)

    # Default for any other long section (detailed_description, etc.)
    return split_by_word_count(text)


def run():
    documents = load_jsonl(CORPUS_PATH)
    print(f"Loaded {len(documents)} documents from corpus")

    all_chunks = []
    section_type_counts = {}

    for doc in documents:
        for section in doc["sections"]:
            section_name = section["section_name"]
            text = section["text"]

            chunk_texts = chunk_section(section_name, text)
            section_type_counts[section_name] = section_type_counts.get(section_name, 0) + len(chunk_texts)

            for i, chunk_text in enumerate(chunk_texts):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc['document_id']}_{section_name}_{i}",
                        "document_id": doc["document_id"],
                        "source_type": doc["source_type"],
                        "source_id": doc["source_id"],
                        "title": doc["title"],
                        "condition_query": doc["condition_query"],
                        "section_name": section_name,
                        "chunk_index": i,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "metadata": doc["metadata"],
                    }
                )

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("\nChunks per section type:")
    for section_name, count in sorted(section_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {section_name}: {count}")

    word_counts = [c["word_count"] for c in all_chunks]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"\nAverage chunk length: {avg_words:.1f} words")
    print(f"Shortest chunk: {min(word_counts)} words")
    print(f"Longest chunk: {max(word_counts)} words")

    print(f"\nChunks saved to: {CHUNKS_PATH}")

    # Show one example of an eligibility criteria chunk split, since that's
    # the most interesting/specific logic in this script.
    example = next((c for c in all_chunks if c["section_name"] == "eligibility_criteria"), None)
    if example:
        print("\n--- Example eligibility criteria chunk ---")
        print(json.dumps(example, indent=2)[:800])


if __name__ == "__main__":
    run()
