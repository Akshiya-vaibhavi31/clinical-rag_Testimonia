"""
config.py

Central place for all project settings: API keys, base URLs, target conditions,
and file paths. Every other script imports from here instead of hardcoding values,
so if something changes (e.g. you add a 4th condition), you only edit it in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env into the environment. If .env doesn't exist yet,
# this silently does nothing (you'll get None for the API key and see an error
# later when we try to use it — that's your signal to go create the .env file).
load_dotenv()

# ---- Secrets / identifiers (from .env) ----
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_TOOL_NAME = os.getenv("NCBI_TOOL_NAME", "clinical-trial-rag-project")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Model name occasionally changes as Google releases new versions — if you
# get a "model not found" error later, check https://ai.google.dev/gemini-api/docs/models
# for the current name and update it here, in one place.
#
# IMPORTANT: gemini-3.6-flash (a newer, higher-tier model) has a very tight
# FREE tier daily quota (~20 requests/day) — fine for single interactive
# questions, but nowhere near enough for a 32-question automated evaluation
# run. gemini-3.5-flash-lite has a much more generous free daily quota
# (roughly 1,000+ requests/day) at slightly lower quality — the right
# tradeoff for bulk/automated use. We use flash-lite everywhere by default;
# swap back to gemini-3.6-flash if you want higher-quality single answers
# and don't mind the tighter daily limit.
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# ---- API base URLs ----
CLINICALTRIALS_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---- Target medical conditions for this project ----
# Chosen because each has hundreds of well-documented, completed trials —
# good for cross-trial comparison questions and rich literature coverage.
TARGET_CONDITIONS = [
    "type 2 diabetes",
    "breast cancer",
    "hypertension",
]

# ---- Dataset size targets (see Phase 1 research for rationale) ----
TRIALS_PER_CONDITION = 200  # ~600 trials total across 3 conditions
ABSTRACTS_PER_CONDITION = 200  # ~600 PubMed abstracts total

# ---- Embedding model settings (Phase 6) ----
# Two options to compare: a fast general-purpose model, and a biomedical-
# specific one. Switch EMBEDDING_MODEL_NAME to try each — everything else
# in the pipeline (embed_corpus.py) adapts automatically.
EMBEDDING_MODELS = {
    "general": "sentence-transformers/all-MiniLM-L6-v2",
    "biomedical": "pritamdeka/S-PubMedBert-MS-MARCO",
}
# Change this key ("general" or "biomedical") to switch which model is used.
ACTIVE_EMBEDDING_MODEL = "general"

# ---- File paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifest"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
