"""
Unit tests for src/chunk_corpus.py.

These test the pure chunking logic in isolation — no network, no
database, no external services. All three chunking strategies (short
section pass-through, eligibility-criteria structural split, generic
word-count splitting) are covered, including the specific edge cases
that caused real bugs earlier in this project (e.g. section boundaries).
"""

import pytest
from src.chunk_corpus import split_by_word_count, split_eligibility_criteria, chunk_section, MAX_CHUNK_WORDS


class TestSplitByWordCount:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short piece of text."
        result = split_by_word_count(text)
        assert result == [text]

    def test_text_exactly_at_limit_returns_single_chunk(self):
        text = " ".join(["word"] * MAX_CHUNK_WORDS)
        result = split_by_word_count(text)
        assert len(result) == 1

    def test_long_text_splits_into_multiple_chunks(self):
        text = " ".join([f"word{i}" for i in range(500)])
        result = split_by_word_count(text, max_words=180, overlap=30)
        assert len(result) > 1
        # Every chunk except possibly the last should respect the max word count
        for chunk in result[:-1]:
            assert len(chunk.split()) == 180

    def test_consecutive_chunks_actually_overlap(self):
        text = " ".join([f"word{i}" for i in range(500)])
        result = split_by_word_count(text, max_words=180, overlap=30)
        first_chunk_words = result[0].split()
        second_chunk_words = result[1].split()
        # The last 30 words of chunk 1 should be the first 30 words of chunk 2
        assert first_chunk_words[-30:] == second_chunk_words[:30]

    def test_empty_text_returns_itself(self):
        result = split_by_word_count("")
        assert result == [""]


class TestSplitEligibilityCriteria:
    def test_splits_on_real_inclusion_exclusion_boundary(self):
        text = (
            "Inclusion Criteria:\n\n"
            + "Adults aged 18 to 64 years. " * 30
            + "\n\nExclusion Criteria:\n\n"
            + "History of cardiac disease. " * 30
        )
        result = split_eligibility_criteria(text)
        # Should produce at least one inclusion chunk and one exclusion chunk
        assert any("Inclusion" in c for c in result)
        assert any("Exclusion" in c for c in result)

    def test_exclusion_chunk_does_not_contain_inclusion_text(self):
        text = "Inclusion Criteria:\nMust be an adult.\n\nExclusion Criteria:\nHistory of cancer."
        result = split_eligibility_criteria(text)
        exclusion_chunks = [c for c in result if "Exclusion" in c]
        assert len(exclusion_chunks) >= 1
        assert "Must be an adult" not in exclusion_chunks[0]

    def test_case_insensitive_boundary_detection(self):
        text = "inclusion criteria: adults only.\n\nEXCLUSION CRITERIA: no minors."
        result = split_eligibility_criteria(text)
        assert len(result) >= 2

    def test_missing_exclusion_boundary_falls_back_to_word_count(self):
        # No "Exclusion Criteria" boundary at all — should not crash,
        # should fall back to generic splitting instead.
        text = "Just inclusion criteria with no exclusion section at all."
        result = split_eligibility_criteria(text)
        assert len(result) >= 1
        assert result[0]  # not empty


class TestChunkSection:
    def test_empty_section_returns_empty_list(self):
        assert chunk_section("brief_summary", "") == []
        assert chunk_section("brief_summary", "   ") == []

    def test_short_section_stays_whole_regardless_of_name(self):
        text = "A short summary of the trial."
        result = chunk_section("detailed_description", text)
        assert result == [text]

    def test_long_eligibility_section_routes_to_structural_split(self):
        text = "Inclusion Criteria:\n" + ("word " * 100) + "\n\nExclusion Criteria:\n" + ("word " * 100)
        result = chunk_section("eligibility_criteria", text)
        # Should have used the structural splitter, not blind word-count on the whole thing
        assert len(result) >= 2

    def test_long_non_eligibility_section_routes_to_word_count_split(self):
        text = " ".join([f"word{i}" for i in range(400)])
        result = chunk_section("detailed_description", text)
        assert len(result) > 1
