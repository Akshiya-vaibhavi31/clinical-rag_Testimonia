"""
preprocess_query.py

Fills the last remaining Phase 9 gap: QUESTION PREPROCESSING.

Two things happen here, before the question ever reaches retrieval:

1. ABBREVIATION EXPANSION: medical abbreviations get expanded to full terms
   (e.g. "T2D" -> "type 2 diabetes"). This matters because our embedding
   model and BM25 index were built on full medical language from real
   trials/papers — a user typing "T2D" might get weaker retrieval than
   someone typing "type 2 diabetes", even though they mean the same thing.

2. TYPO CORRECTION: uses simple fuzzy string matching (via Python's built-in
   difflib, no extra dependency needed) against a vocabulary of known
   medical terms from our corpus. If a word is CLOSE to a known term but not
   an exact match, we correct it. This is a lightweight approach — not a
   real spell-checker with a full dictionary, just enough to catch things
   like "hypertention" -> "hypertension".

Both are deliberately conservative: we only change things we're fairly
confident about, since incorrectly "fixing" a word could hurt retrieval
more than a real typo would.
"""

import difflib
import re

from src.config import TARGET_CONDITIONS

# Common medical abbreviations relevant to our 3 target conditions.
# Deliberately excludes ambiguous ones (e.g. "BC" could mean many things,
# not just "breast cancer") to avoid incorrect expansions.
ABBREVIATION_MAP = {
    "t2d": "type 2 diabetes",
    "t2dm": "type 2 diabetes",
    "dm2": "type 2 diabetes",
    "htn": "hypertension",
    "bp": "blood pressure",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "chf": "congestive heart failure",
    "ckd": "chronic kidney disease",
    "mi": "myocardial infarction",
}

# Vocabulary used for typo correction: our target conditions, common trial
# terminology, and the abbreviation expansions themselves. Kept small and
# specific to this project's domain, rather than a giant general dictionary,
# since we only need to catch typos of terms that actually matter here.
KNOWN_VOCABULARY = set()
for condition in TARGET_CONDITIONS:
    KNOWN_VOCABULARY.update(condition.split())
KNOWN_VOCABULARY.update(ABBREVIATION_MAP.values())
for expansion in ABBREVIATION_MAP.values():
    KNOWN_VOCABULARY.update(expansion.split())
KNOWN_VOCABULARY.update(
    [
        "eligibility",
        "criteria",
        "exclusion",
        "inclusion",
        "trial",
        "trials",
        "dosage",
        "dose",
        "outcomes",
        "adverse",
        "events",
        "phase",
        "patients",
        "patient",
        "treatment",
        "diagnosis",
        "diagnosed",
    ]
)


def expand_abbreviations(question: str) -> tuple[str, list[str]]:
    """
    Replace known abbreviations with their full form. Returns the modified
    question and a list of (abbreviation -> expansion) changes made, so we
    can show the user/log what happened rather than silently rewriting text.
    """
    changes = []
    words = question.split()
    new_words = []

    for word in words:
        # Strip punctuation for matching, but preserve it in the output
        clean_word = re.sub(r"[^\w]", "", word.lower())
        if clean_word in ABBREVIATION_MAP:
            expansion = ABBREVIATION_MAP[clean_word]
            new_words.append(expansion)
            changes.append(f"{clean_word} -> {expansion}")
        else:
            new_words.append(word)

    return " ".join(new_words), changes


def correct_typos(question: str, cutoff: float = 0.82) -> tuple[str, list[str]]:
    """
    For each word not already in our known vocabulary, check if it's a close
    fuzzy match to something that IS in the vocabulary. If so, correct it.

    cutoff=0.82 is deliberately conservative (difflib scores range 0-1) —
    we'd rather miss a real typo than incorrectly "fix" a word that was
    already correct but just isn't in our small vocabulary (e.g. a real
    drug name we don't recognize shouldn't get mangled into an unrelated
    word just because it's unfamiliar).
    """
    changes = []
    words = question.split()
    new_words = []

    for word in words:
        clean_word = re.sub(r"[^\w]", "", word.lower())

        # Skip short words (too easy to false-match) and words already known.
        # Also skip simple singular/plural variants of known words (e.g. if
        # "patients" is known, don't flag "patient" as a typo of it) — this
        # prevents the exact false-positive we hit in testing, where a
        # correctly-spelled singular word got "corrected" into its plural
        # just because only the plural happened to be in our vocabulary.
        is_plural_variant = clean_word + "s" in KNOWN_VOCABULARY
        is_singular_variant = clean_word.endswith("s") and clean_word[:-1] in KNOWN_VOCABULARY

        if len(clean_word) < 5 or clean_word in KNOWN_VOCABULARY or is_plural_variant or is_singular_variant:
            new_words.append(word)
            continue

        matches = difflib.get_close_matches(clean_word, KNOWN_VOCABULARY, n=1, cutoff=cutoff)
        if matches:
            corrected = matches[0]
            new_words.append(corrected)
            changes.append(f"{clean_word} -> {corrected}")
        else:
            new_words.append(word)

    return " ".join(new_words), changes


def preprocess_question(question: str) -> dict:
    """
    Run the full preprocessing pipeline: abbreviation expansion first (since
    expanding "htn" to "hypertension" gives typo correction a cleaner input),
    then typo correction on what remains.
    """
    after_abbrev, abbrev_changes = expand_abbreviations(question)
    after_typo, typo_changes = correct_typos(after_abbrev)

    return {
        "original": question,
        "processed": after_typo,
        "abbreviation_changes": abbrev_changes,
        "typo_changes": typo_changes,
        "was_modified": bool(abbrev_changes or typo_changes),
    }
