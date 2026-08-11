"""
Integration tests for src/database.py.

Uses a real temporary SQLite database (not mocked) — these test genuine
schema creation, constraint enforcement, and multi-table interaction
(foreign keys between queries and answers), which is what makes this an
integration test rather than a unit test: real SQLite is doing real work.
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_db(monkeypatch):
    """Point database.py at a fresh temporary SQLite file for each test."""
    import src.database as db_module

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / "test.db"
        monkeypatch.setattr(db_module, "DB_PATH", temp_path)
        db_module.init_db()
        yield db_module


class TestSchemaCreation:
    def test_init_db_is_idempotent(self, temp_db):
        # Calling init_db() twice should not raise — CREATE TABLE IF NOT EXISTS
        temp_db.init_db()
        temp_db.init_db()

    def test_all_three_tables_exist(self, temp_db):
        with temp_db.get_connection() as conn:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"sources", "queries", "answers"}.issubset(tables)


class TestSourcesTable:
    def test_upsert_then_get_source(self, temp_db):
        temp_db.upsert_source("NCT01057251", "clinical_trial", "Test Trial", condition_query="hypertension")
        result = temp_db.get_source("NCT01057251")
        assert result is not None
        assert result["title"] == "Test Trial"

    def test_get_nonexistent_source_returns_none(self, temp_db):
        assert temp_db.get_source("NCT00000000") is None

    def test_upsert_is_idempotent_and_updates_existing_row(self, temp_db):
        temp_db.upsert_source("NCT01057251", "clinical_trial", "Original Title")
        temp_db.upsert_source("NCT01057251", "clinical_trial", "Updated Title")
        result = temp_db.get_source("NCT01057251")
        assert result["title"] == "Updated Title"
        # Should still be exactly one row, not two
        with temp_db.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM sources WHERE source_id = ?", ("NCT01057251",)).fetchone()[0]
        assert count == 1

    def test_title_cannot_be_null(self, temp_db):
        # The schema declares title NOT NULL — a genuine data-quality gap
        # we found via real corpus data (see build_source_index.py) should
        # still fail loudly here if title is truly missing.
        with pytest.raises(Exception):
            with temp_db.get_connection() as conn:
                conn.execute(
                    "INSERT INTO sources (source_id, source_type, title) VALUES (?, ?, ?)",
                    ("NCT01", "clinical_trial", None),
                )

    def test_search_sources_filters_by_condition(self, temp_db):
        temp_db.upsert_source("NCT01", "clinical_trial", "Trial A", condition_query="hypertension")
        temp_db.upsert_source("NCT02", "clinical_trial", "Trial B", condition_query="type 2 diabetes")
        results = temp_db.search_sources(condition="hypertension")
        assert len(results) == 1
        assert results[0]["source_id"] == "NCT01"


class TestQueryAnswerLogging:
    def test_log_creates_linked_query_and_answer_rows(self, temp_db):
        query_id = temp_db.log_query_and_answer(
            question="What is the eligibility?",
            condition_filter="hypertension",
            low_confidence=False,
            answer_text="Adults 18+.",
            refused=False,
            verification_report={
                "fabricated_count": 0,
                "numeric_mismatch_count": 0,
                "verified_count": 1,
                "total_citations": 1,
            },
            llm_model="gemini-3.5-flash-lite",
            prompt_version="v3",
        )
        assert query_id is not None

        with temp_db.get_connection() as conn:
            answer_row = conn.execute("SELECT * FROM answers WHERE query_id = ?", (query_id,)).fetchone()
        assert answer_row is not None
        assert answer_row["refused"] == 0
        assert answer_row["fabricated_count"] == 0

    def test_get_query_history_returns_most_recent_first(self, temp_db):
        temp_db.log_query_and_answer("First question", None, False, "Answer 1", False, {}, "model", "v1")
        temp_db.log_query_and_answer("Second question", None, False, "Answer 2", False, {}, "model", "v1")

        history = temp_db.get_query_history(limit=10)
        assert len(history) == 2
        assert history[0]["question_text"] == "Second question"  # most recent first

    def test_missing_verification_report_does_not_crash(self, temp_db):
        # A refused answer may have no verification_report at all (None) —
        # this should not raise, should default counts to 0.
        query_id = temp_db.log_query_and_answer(
            question="Unanswerable question",
            condition_filter=None,
            low_confidence=True,
            answer_text="No evidence found.",
            refused=True,
            verification_report=None,
            llm_model="gemini-3.5-flash-lite",
            prompt_version="v3",
        )
        assert query_id is not None
