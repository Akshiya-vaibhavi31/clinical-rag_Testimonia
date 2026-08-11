"""
database.py

Phase 18: Database.

DECISION: SQLite, not PostgreSQL — and 3 tables, not the 8 suggested.

Why SQLite instead of PostgreSQL: this is a single-user local application
with no concurrent-write requirements. PostgreSQL requires a running server
process and separate installation/configuration — real overhead with no
corresponding benefit at this scale. SQLite is a genuine, full relational
SQL database (not a toy) built into Python's standard library — zero extra
dependencies, zero setup, and this schema would port to PostgreSQL with
almost no changes if this project were ever deployed for multiple
concurrent users.

Why only 3 tables (sources, queries, answers), not all 8 suggested:
  - trials, papers, documents, chunks: already well-served by the existing
    JSONL corpus + Chroma vector database (Phases 4-7). Duplicating this
    data into SQL tables would mean keeping two storage systems in sync for
    no new capability.
  - evaluations: already properly handled by Phase 16's experiment tracking
    CSV — a second system logging the same data would be redundant.
  - sources: a GENUINE gap. Every lookup (e.g. the API's /trials/{id}
    endpoint) calls load_all_chunks(), which reads and parses the ENTIRE
    corpus file from disk on every request. A small, indexed SQL table of
    source-level metadata makes structured lookups and filters fast without
    a full file scan every time.
  - queries + answers: also a genuine gap. Every question currently only
    gets appended to a flat JSONL log with no way to query it. A real
    database is exactly the right tool for this kind of structured, growing,
    queryable log.

pgvector investigation (as the phase brief asked): pgvector adds vector
similarity search directly inside PostgreSQL. We deliberately do NOT use
it — our vector search (Chroma, Phase 7) is already built, tested, and
working well at our corpus size. Migrating to pgvector would mean running
a full PostgreSQL server just to get vector search we already have working
for free with Chroma's zero-setup local mode.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "clinical_rag.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL CHECK(source_type IN ('clinical_trial', 'pubmed_abstract')),
    title TEXT NOT NULL,
    condition_query TEXT,
    phase TEXT,
    overall_status TEXT,
    pub_year TEXT,
    doi TEXT,
    pmcid TEXT
);
CREATE INDEX IF NOT EXISTS idx_sources_condition ON sources(condition_query);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);

CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question_text TEXT NOT NULL,
    condition_filter TEXT,
    low_confidence INTEGER NOT NULL CHECK(low_confidence IN (0, 1))
);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp);

CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL REFERENCES queries(id),
    answer_text TEXT NOT NULL,
    refused INTEGER NOT NULL CHECK(refused IN (0, 1)),
    fabricated_count INTEGER DEFAULT 0,
    numeric_mismatch_count INTEGER DEFAULT 0,
    verified_count INTEGER DEFAULT 0,
    total_citations INTEGER DEFAULT 0,
    llm_model TEXT,
    prompt_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_answers_query_id ON answers(query_id);
"""


@contextmanager
def get_connection():
    """Context manager for a SQLite connection — ensures it's always closed properly."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables and indexes if they don't already exist. Safe to call repeatedly."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def upsert_source(
    source_id: str,
    source_type: str,
    title: str,
    condition_query: str = None,
    phase: str = None,
    overall_status: str = None,
    pub_year: str = None,
    doi: str = None,
    pmcid: str = None,
):
    """Insert a source, or update it if the ID already exists (idempotent — safe to re-run)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sources (source_id, source_type, title, condition_query, phase, overall_status, pub_year, doi, pmcid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                title=excluded.title, condition_query=excluded.condition_query,
                phase=excluded.phase, overall_status=excluded.overall_status,
                pub_year=excluded.pub_year, doi=excluded.doi, pmcid=excluded.pmcid
        """,
            (source_id, source_type, title, condition_query, phase, overall_status, pub_year, doi, pmcid),
        )


def get_source(source_id: str):
    """Fast structured lookup by ID — no file scan required."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return dict(row) if row else None


def search_sources(condition: str = None, source_type: str = None, phase: str = None) -> list[dict]:
    """Structured filtering — the kind of query a full JSONL file scan can't do efficiently."""
    query = "SELECT * FROM sources WHERE 1=1"
    params = []
    if condition:
        query += " AND condition_query = ?"
        params.append(condition)
    if source_type:
        query += " AND source_type = ?"
        params.append(source_type)
    if phase:
        query += " AND phase LIKE ?"
        params.append(f"%{phase}%")

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def log_query_and_answer(
    question: str,
    condition_filter: str,
    low_confidence: bool,
    answer_text: str,
    refused: bool,
    verification_report: dict,
    llm_model: str,
    prompt_version: str,
) -> int:
    """
    Log one question + its answer as linked rows. Returns the new query's ID.
    """
    report = verification_report or {}
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO queries (timestamp, question_text, condition_filter, low_confidence)
            VALUES (?, ?, ?, ?)
        """,
            (datetime.now(timezone.utc).isoformat(), question, condition_filter, int(low_confidence)),
        )
        query_id = cursor.lastrowid

        conn.execute(
            """
            INSERT INTO answers (query_id, answer_text, refused, fabricated_count,
                                  numeric_mismatch_count, verified_count, total_citations,
                                  llm_model, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                query_id,
                answer_text,
                int(refused),
                report.get("fabricated_count", 0),
                report.get("numeric_mismatch_count", 0),
                report.get("verified_count", 0),
                report.get("total_citations", 0),
                llm_model,
                prompt_version,
            ),
        )

        return query_id


def get_query_history(limit: int = 50) -> list[dict]:
    """Recent question/answer history, joined into one readable view."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT q.id, q.timestamp, q.question_text, q.condition_filter, q.low_confidence,
                   a.answer_text, a.refused, a.fabricated_count, a.numeric_mismatch_count,
                   a.llm_model, a.prompt_version
            FROM queries q
            JOIN answers a ON a.query_id = q.id
            ORDER BY q.timestamp DESC
            LIMIT ?
        """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
