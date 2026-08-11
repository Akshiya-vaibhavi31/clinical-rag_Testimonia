"""
Unit tests for src/hybrid_retrieval.py's pure logic functions.

NOTE: this module imports chromadb, rank_bm25, and sentence_transformers
at the top level. If any of those aren't installed, this whole file will
fail to collect — pytest.importorskip handles that gracefully rather than
erroring the whole test run.
"""

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")

from src.hybrid_retrieval import tokenize, reciprocal_rank_fusion, RRF_K


class TestTokenize:
    def test_lowercases_and_splits_on_word_boundaries(self):
        assert tokenize("Hypertension Trial 2024") == ["hypertension", "trial", "2024"]

    def test_strips_punctuation(self):
        assert tokenize("metformin, 500mg!") == ["metformin", "500mg"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_preserves_alphanumeric_ids(self):
        assert "nct01057251" in tokenize("Trial NCT01057251 results")


class TestReciprocalRankFusion:
    def test_chunk_ranked_first_in_both_lists_scores_highest(self):
        semantic = [{"chunk_id": "A", "rank": 1}, {"chunk_id": "B", "rank": 2}]
        keyword = [{"chunk_id": "A", "rank": 1}, {"chunk_id": "C", "rank": 2}]
        fused = reciprocal_rank_fusion(semantic, keyword)
        assert fused["A"] > fused["B"]
        assert fused["A"] > fused["C"]

    def test_score_matches_rrf_formula(self):
        semantic = [{"chunk_id": "A", "rank": 1}]
        keyword = []
        fused = reciprocal_rank_fusion(semantic, keyword)
        expected = 1 / (RRF_K + 1)
        assert fused["A"] == pytest.approx(expected)

    def test_chunk_only_in_one_list_still_gets_a_score(self):
        semantic = [{"chunk_id": "A", "rank": 1}]
        keyword = []
        fused = reciprocal_rank_fusion(semantic, keyword)
        assert "A" in fused
        assert fused["A"] > 0

    def test_empty_lists_produce_empty_result(self):
        assert reciprocal_rank_fusion([], []) == {}

    def test_lower_rank_number_scores_higher(self):
        # rank 1 (best) should score higher than rank 10 (worse)
        semantic = [{"chunk_id": "best", "rank": 1}, {"chunk_id": "worst", "rank": 10}]
        fused = reciprocal_rank_fusion(semantic, [])
        assert fused["best"] > fused["worst"]
