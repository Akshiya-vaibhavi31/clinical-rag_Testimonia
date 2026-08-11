"""
Integration tests for retrieval + generation integration in
src/rag_pipeline.py's answer_question().

These mock the two genuinely external boundaries (the embedding-based
retrieval, and the Gemini API call) while exercising everything in
between for real: prompt construction, evidence assembly, citation
verification, and the tiered refuse/caveat logic. This is what makes it
an integration test rather than a unit test — several real internal
components genuinely work together here, only the network edges are
stubbed out.

NOTE: requires `google-genai` to be importable (even though we never make
a real network call — the import itself must succeed).
"""

import pytest

pytest.importorskip("google.genai")

from unittest.mock import patch, MagicMock


def make_fake_gemini_response(text: str):
    """Build a fake object matching the shape of a real Gemini response."""
    fake = MagicMock()
    fake.text = text
    return fake


SAMPLE_EVIDENCE_CHUNK = {
    "chunk_id": "trial_NCT01057251_brief_summary_0",
    "source_id": "NCT01057251",
    "source_type": "clinical_trial",
    "title": "Nebivolol in Patients With Systolic Stage 2 Hypertension",
    "section_name": "brief_summary",
    "text": "The primary object of this study is to evaluate the efficacy of nebivolol.",
    "metadata": {"phases": ["PHASE4"], "overall_status": "COMPLETED"},
}


class TestRetrievalGenerationIntegration:
    def test_no_evidence_found_refuses_without_calling_llm(self):
        from src.rag_pipeline import answer_question

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={"chunks": [], "low_confidence": True, "top_semantic_score": 0.0},
        ):
            with patch("src.rag_pipeline.genai.Client") as mock_client_class:
                result = answer_question("What is a completely unrelated question?", verbose=False)

        assert result["refused"] is True
        assert result["evidence_chunks"] == []
        # The LLM should never even be called if there's no evidence at all
        mock_client_class.assert_not_called()

    def test_clean_answer_flows_through_correctly(self):
        from src.rag_pipeline import answer_question

        fake_response = make_fake_gemini_response("The study evaluated nebivolol efficacy [source: NCT01057251].")

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [SAMPLE_EVIDENCE_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.8,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**SAMPLE_EVIDENCE_CHUNK, "rerank_score": 5.0}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client_class:
                    mock_client_class.return_value.models.generate_content.return_value = fake_response
                    result = answer_question("What did the trial evaluate?", verbose=False)

        assert result["refused"] is False
        assert "NCT01057251" in result["cited_source_ids"]
        assert result["verification_report"]["passed"] is True

    def test_fabricated_citation_triggers_regeneration_then_hard_refusal(self):
        from src.rag_pipeline import answer_question

        # First response fabricates a source; second (regeneration) attempt does too
        bad_response = make_fake_gemini_response("Fake finding [source: NCT99999999].")

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [SAMPLE_EVIDENCE_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.8,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**SAMPLE_EVIDENCE_CHUNK, "rerank_score": 5.0}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client_class:
                    mock_client_class.return_value.models.generate_content.return_value = bad_response
                    result = answer_question("What did the trial evaluate?", verbose=False)

        assert result["refused"] is True
        assert "can't provide a verified answer" in result["answer_text"].lower()

    def test_numeric_mismatch_only_gets_caveat_not_hard_refusal(self):
        from src.rag_pipeline import answer_question

        # A real source, but a number not present in the evidence
        wrong_number_response = make_fake_gemini_response("The study reported a 99% improvement [source: NCT01057251].")

        with patch(
            "src.rag_pipeline.hybrid_search_raw",
            return_value={
                "chunks": [SAMPLE_EVIDENCE_CHUNK],
                "low_confidence": False,
                "top_semantic_score": 0.8,
            },
        ):
            with patch("src.rag_pipeline.rerank_chunks", return_value=[{**SAMPLE_EVIDENCE_CHUNK, "rerank_score": 5.0}]):
                with patch("src.rag_pipeline.genai.Client") as mock_client_class:
                    # Both the original AND the regeneration attempt keep the bad number,
                    # so this should land in the "numeric mismatch only" caveat path.
                    mock_client_class.return_value.models.generate_content.return_value = wrong_number_response
                    result = answer_question("What improvement did the study report?", verbose=False)

        # Should NOT be a hard refusal — fabrication_count is 0, only a numeric mismatch
        assert result["refused"] is False
        assert "could not be independently confirmed against the retrieved evidence" in result["answer_text"]
