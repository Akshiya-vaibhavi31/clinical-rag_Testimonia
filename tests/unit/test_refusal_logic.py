"""
Unit tests for src/rag_pipeline.py's detect_refusal().

This function was extracted from inline code inside answer_question()
specifically to make it unit-testable — refusal detection is a heuristic
(documented as such) that has needed several real fixes as Gemini's
phrasing varied across runs, so it deserves direct test coverage.

NOTE: rag_pipeline.py imports google.genai at the top level. If that
package isn't installed, this file is skipped rather than failing the
whole test run.
"""

import pytest

pytest.importorskip("google.genai")

from src.rag_pipeline import detect_refusal


class TestDetectRefusal:
    def test_standard_insufficient_evidence_phrase(self):
        text = "The retrieved sources do not contain sufficient evidence to answer this question."
        assert detect_refusal(text) is True

    def test_hard_refusal_after_failed_regeneration(self):
        text = "I can't provide a verified answer to this question. The generated response contained citations..."
        assert detect_refusal(text) is True

    def test_genuine_answer_is_not_flagged_as_refusal(self):
        text = "The trial reported a 12% reduction in blood pressure among patients [source: NCT01057251]."
        assert detect_refusal(text) is False

    def test_personal_opinion_decline_variant_1(self):
        text = "I am an AI assistant and do not have personal opinions or beliefs."
        assert detect_refusal(text) is True

    def test_personal_opinion_decline_variant_2(self):
        # A real second wording Gemini used for the same correct behavior —
        # exact phrase matching alone missed this one originally.
        text = "I am a medical evidence retrieval assistant and cannot provide personal opinions or determine which diabetes drug is the best."
        assert detect_refusal(text) is True

    def test_personal_opinion_decline_variant_3(self):
        # A third distinct wording for the same behavior, from a later run.
        text = "I do not have personal opinions or thoughts, as I am a medical evidence retrieval assistant."
        assert detect_refusal(text) is True

    def test_answer_with_numbers_is_not_falsely_flagged(self):
        # Regression guard: a real answer full of citations/numbers should
        # never accidentally match the personal-opinion regex.
        text = "Results showed RR=0.77 (95% CI 0.70-0.86) for the combined group [source: PMID 42572726]."
        assert detect_refusal(text) is False

    def test_case_insensitivity(self):
        text = "THE RETRIEVED SOURCES DO NOT CONTAIN SUFFICIENT EVIDENCE TO ANSWER THIS QUESTION."
        assert detect_refusal(text) is True

    def test_partial_refusal_with_real_data_is_still_flagged(self):
        # A partial refusal (declines the specific question but offers
        # related real data) should still be recognized as a refusal of
        # the original question.
        text = "The retrieved sources do not contain information regarding an overall mortality rate, however individual studies report specific rates."
        assert detect_refusal(text) is True
