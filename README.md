# Testimonia — Clinical Trial & Medical Literature RAG Assistant

A citation-grounded question-answering system over real clinical trial
records and published medical literature. Every answer is traced to a
specific, inspectable source — or the system explicitly refuses to answer
rather than guessing.

**This is a research/evidence-retrieval tool, not a medical advisor.** It
does not diagnose, prescribe, or give personalized medical advice.

## What it does

Ask a question like *"What are common exclusion criteria for hypertension
trials?"* and the system:
1. Retrieves the most relevant evidence from ~1,100 indexed clinical trials
   and PubMed abstracts (hypertension, type 2 diabetes, breast cancer)
2. Generates an answer using only that retrieved evidence
3. Independently verifies every citation the answer makes — checking that
   cited sources are real and that any numeric claims are actually
   supported by the evidence
4. Refuses to answer, or flags a caveat, when the evidence doesn't
   genuinely support a confident claim

## Real evaluation results

Measured across a 32-question evaluation set covering fact lookup,
eligibility, trial comparison, outcome extraction, adverse events, study
design, multi-document synthesis, and deliberately unanswerable/adversarial
questions:

| Metric | Result |
|---|---|
| Overall refusal decision accuracy | **90.6%** |
| Refusal recall (catches things it should refuse) | **100%** |
| Refusal precision | 75% |
| Citation fabrication rate | **0.0%** — held across every evaluation run |
| Faithfulness (citations genuinely supported by evidence) | 75.9% |
| Retrieval recall@5 | 87% |

See `data/eval/README.md` and `data/eval/eval_set.json` for the full
evaluation methodology, and `src/evaluation_metrics.py` for how each
metric is computed (and its documented limitations).

## Architecture

```
User question
    │
    ▼
Query preprocessing (typo correction, abbreviation expansion)
    │
    ▼
Hybrid retrieval — semantic search (sentence-transformers) + BM25 keyword
search, fused via Reciprocal Rank Fusion, plus explicit NCT/PMID injection
when a source is named directly in the question
    │
    ▼
Cross-encoder reranking (with pinning: an explicitly-named source is
guaranteed to survive reranking, capped at 3 chunks per source)
    │
    ▼
LLM generation (Gemini) — grounded strictly in retrieved evidence, with
explicit rules against filling gaps from general knowledge and against
following any instructions embedded in the evidence itself
    │
    ▼
Citation verification — checks every cited source is real, and every
numeric claim is backed by the evidence; triggers one corrective
regeneration attempt on failure
    │
    ▼
Structured answer + citations + evidence viewer
```

## Tech stack

- **Data sources:** ClinicalTrials.gov API v2, PubMed E-utilities
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector search:** Chroma (local, zero-setup)
- **Reranking:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM:** Google Gemini (flash-lite tier)
- **Backend:** FastAPI, with Pydantic validation, rate limiting (slowapi),
  structured logging
- **Database:** SQLite — a lightweight index over sources plus a queryable
  log of every question/answer (see `src/database.py` for why SQLite was
  chosen over PostgreSQL at this project's scale)
- **Frontend:** a single self-contained HTML/CSS/JS file (no build step)

## Project structure

```
src/
  clients/                    # ClinicalTrials.gov + PubMed API clients
  ingest_trials.py, ingest_pubmed.py, explore_data.py
  build_corpus.py, chunk_corpus.py, embed_corpus.py, build_vector_db.py
  hybrid_retrieval.py, reranker.py, preprocess_query.py
  rag_pipeline.py             # the core pipeline + system prompt
  citation_verification.py    # fabrication + numeric claim checking
  citations.py, inspect_citation.py
  database.py, build_source_index.py
  evaluation_metrics.py, run_evaluation.py, experiment_tracking.py
  test_gemini.py, test_prompt_injection.py
  api/                        # FastAPI backend
data/
  eval/                       # evaluation question set + results
frontend/
  index.html                  # the web app
SECURITY.md                   # security & privacy review
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
cp .env.example .env
# edit .env with your real NCBI and Gemini API keys
```

### Build the data pipeline 

```bash
python -m src.ingest_trials
python -m src.ingest_pubmed
python -m src.build_corpus
python -m src.chunk_corpus
python -m src.embed_corpus
python -m src.build_vector_db
python -m src.build_source_index
```

### Run the backend

```bash
uvicorn src.api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Run the frontend

Open `frontend/index.html` with a local server 
```bash
cd frontend && python3 -m http.server 5500
```
The backend must be running first — the frontend calls
`http://127.0.0.1:8000` 

### Run the evaluation

```bash
python -m src.run_evaluation
```

### Verify prompt injection resistance

```bash
python -m src.test_prompt_injection
```

## Key design decisions

- **Hybrid retrieval over pure semantic search** — BM25 catches exact drug
  names and IDs that embedding similarity alone can miss.
- **Explicit ID injection** — a trial's own NCT ID is never part of its
  searchable text, so a question naming a trial directly would otherwise
  never find it through similarity search. Found via evaluation, fixed by
  guaranteeing named sources are retrieved and survive reranking.
- **Citation verification with tiered severity** — a fabricated *source* is
  always a hard refusal (0% tolerance). An unconfirmed *number* in an
  otherwise well-sourced answer gets a caveat instead of a blanket refusal,
  since evaluation showed the numeric-matching heuristic has a real
  false-positive rate on dense, multi-source claims.
- **SQLite over PostgreSQL, 3 tables not 8** — chosen to match this
  project's actual scale (single local user) rather than over-building
  infrastructure. See `src/database.py` for the full reasoning.

## Known limitations 

- Numeric claim verification is lexical/heuristic, not true semantic
  entailment — it can produce false positives on claims involving
  timeframes or multi-part figures.
- No backend authentication — appropriate for local single-user use, not
  for shared/public deployment. See `SECURITY.md`.
- The frontend's login screen is a UI demonstration only, not connected to
  real backend authentication.
- Retrieval recall (~87%) is the primary bottleneck on answer accuracy —
  higher-recall retrieval would likely improve overall accuracy further.

## Security

See `SECURITY.md` for the full security and privacy review, including
verified protection against SQL injection and prompt injection through
retrieved documents (tested live against a real adversarial example).

## Debugging highlights

This project surfaced and fixed a number of real bugs through iterative
testing — not hypothetical edge cases, but ones caught by actually running
the system against real data:
- A doubled "NCT" prefix in citation formatting
- DOI/PMCID fields being pulled from a paper's reference list instead of
  its own metadata (an XML-scoping bug)
- A regex that silently failed to parse multi-source citation brackets,
  making a fabricated citation invisible to the verifier
- A false-positive explosion in numeric verification on multi-document
  synthesis answers, traced to unscoped number-matching across shared
  citation groups

Each of these was found by building an evaluation harness and actually
running it — not by inspection alone.
