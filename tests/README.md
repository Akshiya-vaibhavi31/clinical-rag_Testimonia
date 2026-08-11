# Test Suite

Real, structured tests covering unit, integration, and end-to-end levels,
as required by Phase 21.

## How to run everything

```bash
pip install -r requirements.txt   # adds pytest and httpx
pytest tests/ -v
```

Run just one level:
```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

## Structure

```
tests/
├── unit/
│   ├── test_chunking.py           # chunk_corpus.py's splitting logic
│   ├── test_parsing.py            # PubMed XML parsing (DOI/PMCID bug regression)
│   ├── test_retrieval.py          # BM25 tokenization + Reciprocal Rank Fusion math
│   ├── test_citation_mapping.py   # citation extraction + verification
│   └── test_refusal_logic.py      # detect_refusal() heuristic
├── integration/
│   ├── test_api.py                # FastAPI endpoints, real request/response cycle
│   ├── test_database.py           # real SQLite schema, constraints, multi-table logic
│   └── test_retrieval_generation.py  # retrieval + reranking + verification working together
└── e2e/
    └── test_full_pipeline.py      # complete question → answer flow, including failure cases
```

## An honest note on how this was built and verified

This suite was written and built without a live coding environment with
every project dependency installed (no `google-genai`, `fastapi`,
`chromadb`, `rank-bm25`, or `tenacity` were available). Rather than write
untested test code and hope it's correct, every test was handled one of
two ways:

1. **Modules with no external dependencies** (`chunk_corpus.py`,
   `citation_verification.py`, `database.py` — only need `re`, `json`,
   `sqlite3`) were tested for real: every single assertion in
   `test_chunking.py`, `test_citation_mapping.py`, and `test_database.py`
   was actually executed against the real project code and confirmed
   passing (48 assertions, 0 failures) before being finalized here.

2. **Modules requiring unavailable packages** (`test_parsing.py`,
   `test_retrieval.py`, `test_refusal_logic.py`, and everything in
   `integration/` and `e2e/`) were written against the real function
   signatures and exact string literals (verified via direct inspection
   of the source files, not assumption), and their underlying logic was
   independently verified by faithfully replicating the relevant code and
   running the same assertions against that replica.

**What this means for you:** these tests should pass immediately in your
real environment (where every dependency is installed), since they were
written against verified real signatures and behavior — but you should
still run `pytest tests/ -v` yourself as the final confirmation, the same
way every other phase in this project was only considered "done" once
actually run for real, not just written.

## Packages required per test file

Some test files use `pytest.importorskip(...)` to gracefully skip
themselves (rather than error the whole run) if an optional dependency
isn't installed:

| File | Requires |
|---|---|
| `test_parsing.py` | `tenacity` |
| `test_retrieval.py` | `chromadb`, `rank-bm25` |
| `test_refusal_logic.py`, `test_retrieval_generation.py`, `test_full_pipeline.py` | `google-genai` |
| `test_api.py` | `fastapi` |

Installing everything in `requirements.txt` covers all of these.

## A real refactor this phase produced

Writing `test_refusal_logic.py` surfaced a genuine code-quality gap: the
refusal-detection heuristic was embedded inline inside `answer_question()`
in `rag_pipeline.py`, making it impossible to unit test in isolation. It's
now extracted into a standalone `detect_refusal()` function — the same
logic, just properly testable. This is a good example of tests improving
code structure, not just checking it.
