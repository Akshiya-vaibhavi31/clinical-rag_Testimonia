# Running with Docker

## Why Docker is genuinely appropriate here (not just checking a box)

This project already hit a real "works differently depending on
environment" bug during Phase 21 — `pytest` resolved to a completely
different Python installation (Anaconda) than the one with all the
project's packages installed (`venv`), causing confusing import errors
that had nothing to do with the actual code. Docker exists specifically to
eliminate this entire category of problem: the container always has
exactly the right Python version and exactly the right packages,
regardless of whatever else is installed on the host machine.

## What's containerized

Just **one container**, not two — since the frontend is now served
directly by the FastAPI backend itself (see the "merge frontend and
backend" change), there's no separate frontend service needed.

**Data is deliberately NOT baked into the image.** Your corpus, vector
database, embeddings, and SQLite database are mounted in from your local
`data/` folder instead. This keeps the image small, keeps your data
persistent across rebuilds, and avoids ever needing real API keys just to
build an image (secrets are provided at *runtime* via `.env`, per Phase
20's security review).

## First-time setup

```bash
# 1. Make sure your .env file exists with real API keys
cp .env.example .env
# edit .env with your real NCBI_API_KEY and GEMINI_API_KEY

# 2. Build the image
docker compose build

# 3. Start the container
docker compose up
```

Visit `http://localhost:8000/` — the frontend loads directly, no separate
server needed.

## If you already have data built locally (your situation)

Since you've already run the ingestion/chunking/embedding pipeline in
this project, your `data/` folder already has everything the app needs.
Because `docker-compose.yml` mounts `./data:/app/data`, that existing data
is automatically visible inside the container — **no rebuilding needed.**
Just `docker compose up` and it should work immediately with your real
data.

## On a genuinely fresh machine (no data built yet)

This is the actual test of "runs consistently on another machine." Someone
cloning this repo fresh would need to build the data pipeline once, which
can be done by running each script *inside* the container:

```bash
docker compose run api python -m src.ingest_trials
docker compose run api python -m src.ingest_pubmed
docker compose run api python -m src.build_corpus
docker compose run api python -m src.chunk_corpus
docker compose run api python -m src.embed_corpus
docker compose run api python -m src.build_vector_db
docker compose run api python -m src.build_source_index
```

Since the volume mount means this writes directly to the host's `data/`
folder (not just inside a throwaway container), this only needs to be done
once — after that, `docker compose up` alone is enough from then on.

## Checking it's actually working

```bash
curl http://localhost:8000/health
```

Should return `{"status": "ok", "corpus_loaded": true, "vector_db_ready": true}`.

## Stopping

```bash
docker compose down
```

Your data in `./data` is untouched — only the container itself is removed,
not your actual corpus/database.

## An honest limitation of this review

I don't have Docker available in the environment I built this in, so I
could not literally run `docker compose build` / `up` myself to confirm
this works end-to-end. I validated the `docker-compose.yml` file's YAML
syntax directly and reasoned carefully through each choice (layer caching
order, volume mounts, why data isn't baked in), but **you should run this
yourself** as the real confirmation — the same standard every other phase
in this project has been held to.
