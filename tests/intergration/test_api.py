"""
Integration tests for src/api/main.py.

Uses FastAPI's TestClient to make real HTTP-style requests against the
app object directly (no network, no running server needed) — this is a
genuine integration test because it exercises the full request/response
cycle: routing, Pydantic validation, the endpoint handler, and response
serialization all together.

The expensive external calls (Gemini generation, embedding model) are
mocked at the answer_question/hybrid_search_raw boundary — this keeps the
test fast, free, and deterministic, while still genuinely testing how the
API layer handles both success and failure from that boundary.

NOTE: requires `fastapi` to be installed to run.
"""

import pytest

pytest.importorskip("fastapi")

from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSourceEndpoints:
    def test_trial_with_invalid_prefix_returns_400(self, client):
        response = client.get("/trials/12345")  # missing "NCT" prefix
        assert response.status_code == 400

    def test_paper_with_non_numeric_id_returns_400(self, client):
        response = client.get("/papers/not-a-number")
        assert response.status_code == 400

    def test_nonexistent_source_returns_404(self, client):
        with patch("src.api.main.get_chunks_for_explicit_ids", return_value=[]):
            with patch("src.api.main.get_db_source", return_value=None):
                response = client.get("/trials/NCT00000000")
        assert response.status_code == 404


class TestAskEndpointValidation:
    def test_blank_question_returns_422(self, client):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_too_short_question_returns_422(self, client):
        response = client.post("/ask", json={"question": "hi"})  # below min_length
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422


class TestAskEndpointWithMockedPipeline:
    def test_successful_answer_returns_200_with_expected_shape(self, client):
        mock_result = {
            "answer_text": "The trial found X [source: NCT01057251].",
            "evidence_chunks": [
                {
                    "chunk_id": "trial_NCT01057251_brief_summary_0",
                    "source_id": "NCT01057251",
                    "source_type": "clinical_trial",
                    "title": "Test Trial",
                    "section_name": "brief_summary",
                    "text": "...",
                    "metadata": {},
                }
            ],
            "cited_source_ids": ["NCT01057251"],
            "refused": False,
            "verification_report": {
                "total_citations": 1,
                "verified_count": 1,
                "fabricated_count": 0,
                "numeric_mismatch_count": 0,
                "passed": True,
            },
            "low_confidence": False,
            "regenerated": False,
        }
        with patch("src.api.main.answer_question", return_value=mock_result):
            with patch("src.api.main.log_query_and_answer"):
                response = client.post("/ask", json={"question": "What did the trial find?"})
        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False
        assert len(body["citations"]) == 1

    def test_refused_answer_returns_200_with_refused_true(self, client):
        mock_result = {
            "answer_text": "The retrieved sources do not contain sufficient evidence to answer this question.",
            "evidence_chunks": [],
            "cited_source_ids": [],
            "refused": True,
            "verification_report": None,
            "low_confidence": True,
            "regenerated": False,
        }
        with patch("src.api.main.answer_question", return_value=mock_result):
            with patch("src.api.main.log_query_and_answer"):
                response = client.post("/ask", json={"question": "What causes an unrelated rare disease?"})
        assert response.status_code == 200
        assert response.json()["refused"] is True

    def test_pipeline_exception_returns_500_not_a_crash(self, client):
        with patch("src.api.main.answer_question", side_effect=RuntimeError("boom")):
            response = client.post("/ask", json={"question": "A perfectly valid question here"})
        assert response.status_code == 500
        # The real error message must never leak to the client
        assert "boom" not in response.text


class TestCompareEndpointValidation:
    def test_single_source_id_returns_422(self, client):
        response = client.post("/compare", json={"source_ids": ["NCT01057251"]})
        assert response.status_code == 422

    def test_two_source_ids_passes_validation(self, client):
        mock_result = {
            "answer_text": "Comparison text.",
            "evidence_chunks": [],
            "cited_source_ids": [],
            "refused": False,
            "verification_report": None,
            "low_confidence": False,
            "regenerated": False,
        }
        with patch("src.api.main.answer_question", return_value=mock_result):
            response = client.post("/compare", json={"source_ids": ["NCT01057251", "NCT00728858"]})
        assert response.status_code == 200
