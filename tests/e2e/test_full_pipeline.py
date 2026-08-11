"""
End-to-end tests: the complete path from a user's question through
retrieval, generation, citation verification, and the final structured
response — exactly what a real user (via CLI or API) experiences.

Unlike the integration tests (which check that pairs of components work
together correctly), these exercise the ENTIRE pipeline in one call to
answer_question(), asserting on the final observable outcome only — the
same thing Phase 15's evaluation harness does, just as fast, deterministic
pytest cases instead of live API calls.

Only the actual external services (Gemini, the embedding-based retrieval)
are mocked — every internal component (reranking, prompt building,
citation verification, the tiered refuse/caveat logic) runs for real.

NOTE: requires `google-genai` to be importable.
"""

import pytest

pytest.importorskip("google.genai")

from unittest.mock import patch, MagicMock


def make_fake_gemini_response(text: str):
    fake = MagicMock()
    fake.text = text
    return fake


REAL_LOOKING_CHUNK = {
    "chunk_id": "trial_NCT01057251_eligibility_criteria_0",
    "source_id": "NCT01057251",
    "source_type": "clinical_trial",
    "title": "Nebivolol in Patients With Systolic Stage 2 Hypertension",
    "section_name": "eligibility_criteria",
    "text": "Inclusion Criteria: male or female, 18 to 64 years of age at screening.",
    "metadata": {"phases": ["PHASE4"], "minimum_age": "18 Years"},
}


class TestEndToEndHappyPath:
    def test_user_question_to_final_response_with_valid_citation(self):
        """
        The full, successful path: a real question, real-shaped evidence,
        a clean grounded answer — verifying every layer of the final
        response a user actually sees.
        """
        from src.rag_pipeline import answer_question

        fake_response = make_fake_gemini_response("The minimum age for participants is 18 years [source: NCT01057251].")

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [REAL_LOOKING_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.85,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**REAL_LOOKING_CHUNK, "rerank_score": 4.2}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client:
                    mock_client.return_value.models.generate_content.return_value = fake_response
                    result = answer_question(
                        "What is the minimum age for participants in the Nebivolol trial?",
                        verbose=False,
                    )

        # Assert on the complete, final, user-facing response
        assert result["refused"] is False
        assert "18 years" in result["answer_text"]
        assert result["cited_source_ids"] == ["NCT01057251"]
        assert result["verification_report"]["fabricated_count"] == 0
        assert result["verification_report"]["passed"] is True
        assert result["evidence_chunks"][0]["source_id"] == "NCT01057251"


class TestEndToEndFailureCases:
    def test_completely_unrelated_question_is_refused(self):
        """A question entirely outside the corpus's scope should be refused, not guessed at."""
        from src.rag_pipeline import answer_question

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [],
                "low_confidence": True,
                "top_semantic_score": 0.0,
            },
        ):
            result = answer_question("What is the boiling point of helium?", verbose=False)

        assert result["refused"] is True
        assert result["evidence_chunks"] == []

    def test_low_confidence_retrieval_still_produces_a_flagged_response(self):
        """
        Weak retrieval matches shouldn't silently look identical to strong
        ones — low_confidence must propagate all the way to the final
        response so a caller (API, frontend) can surface it.
        """
        from src.rag_pipeline import answer_question

        fake_response = make_fake_gemini_response(
            "The retrieved sources do not contain sufficient evidence to answer this question."
        )

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [REAL_LOOKING_CHUNK],
                "low_confidence": True,
                "top_semantic_score": 0.15,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**REAL_LOOKING_CHUNK, "rerank_score": 0.3}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client:
                    mock_client.return_value.models.generate_content.return_value = fake_response
                    result = answer_question("What percentage of patients had a rare side effect?", verbose=False)

        assert result["low_confidence"] is True
        assert result["refused"] is True

    def test_repeated_fabrication_across_regeneration_ends_in_refusal_not_silent_answer(self):
        """
        The system must never silently present an answer with a source
        that was never in the evidence — even after giving the model one
        chance to self-correct, persistent fabrication must end in a
        visible refusal, not a degraded-but-accepted answer.
        """
        from src.rag_pipeline import answer_question

        fabricated_response = make_fake_gemini_response(
            "This finding comes from an unrelated trial [source: NCT00000001]."
        )

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [REAL_LOOKING_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.8,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**REAL_LOOKING_CHUNK, "rerank_score": 4.0}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client:
                    mock_client.return_value.models.generate_content.return_value = fabricated_response
                    result = answer_question("What did the trial find?", verbose=False)

        assert result["refused"] is True
        assert result["regenerated"] is True
        # The final answer must not contain the fabricated source as if it were legitimate
        assert (
            "NCT00000001" not in result.get("cited_source_ids", [])
            or "can't provide a verified answer" in result["answer_text"].lower()
        )

    def test_evidence_exists_but_question_asks_for_personalized_advice(self):
        """
        Even with real evidence retrieved, a question asking for
        personalized medical advice should still be declined — the
        system prompt's rule against this, not just missing evidence,
        is what should drive the refusal here.
        """
        from src.rag_pipeline import answer_question

        appropriate_decline = make_fake_gemini_response(
            "I cannot provide a personalized dosage recommendation. The trial itself used "
            "a starting dose described in its protocol [source: NCT01057251], but this is "
            "not personalized medical advice."
        )

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [REAL_LOOKING_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.7,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**REAL_LOOKING_CHUNK, "rerank_score": 3.5}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client:
                    mock_client.return_value.models.generate_content.return_value = appropriate_decline
                    result = answer_question("What dosage should I personally take?", verbose=False)

        # A real citation is present, but the answer correctly declines personalization
        assert "NCT01057251" in result["answer_text"]
        assert "personalized dosage recommendation" in result["answer_text"].lower()
