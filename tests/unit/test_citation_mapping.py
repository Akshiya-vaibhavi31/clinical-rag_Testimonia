"""
Unit tests for src/citation_verification.py.

These specifically cover the real bugs found and fixed during this
project's development — multi-source citation brackets, the trailing-
punctuation number-extraction bug, and the grouping fix for shared
sentences — not just the happy path.
"""

import pytest
from src.citation_verification import extract_citations, extract_numbers, tokenize_words, verify_citations


class TestExtractCitations:
    def test_single_citation_extracted(self):
        text = "The trial found X [source: NCT01057251]."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0][1] == "NCT01057251"

    def test_pmid_citation_normalized_with_prefix(self):
        text = "The paper found Y [source: PMID 42572726]."
        citations = extract_citations(text)
        assert citations[0][1] == "PMID 42572726"

    def test_multi_id_single_bracket_extracts_all_ids(self):
        # Regression test for a real bug: the original regex required one
        # ID per bracket and silently matched NOTHING when the model wrote
        # several IDs in a single bracket — including a fabricated ID that
        # went completely unverified as a result.
        text = "[source: NCT04434924, source: NCT03683069, source: NCT047251]"
        citations = extract_citations(text)
        ids = [c[1] for c in citations]
        assert len(ids) == 3
        assert "NCT04434924" in ids
        assert "NCT03683069" in ids
        assert "NCT047251" in ids  # the "fabricated" one must still be extracted to be checkable

    def test_citations_in_same_sentence_share_identical_segment_text(self):
        # Regression test: citations in one sentence must produce the SAME
        # segment string so verify_citations can correctly group them.
        text = "Ages ranged from 18-64 [source: NCT01057251] to 20-90 [source: NCT00728858]."
        citations = extract_citations(text)
        segments = [c[0] for c in citations]
        assert segments[0] == segments[1]

    def test_no_citations_returns_empty_list(self):
        assert extract_citations("No citations here at all.") == []


class TestExtractNumbers:
    def test_extracts_plain_integer(self):
        assert "23" in extract_numbers("Reduced risk by 23%.")

    def test_extracts_decimal(self):
        assert "1.5" in extract_numbers("A dose of 1.5 mg.")

    def test_trailing_period_does_not_get_absorbed_into_number(self):
        # Regression test for a real bug: "AC2993." at the end of a
        # sentence used to extract as "2993." (with the period attached),
        # which didn't match "2993" extracted elsewhere without a period —
        # causing a false numeric-mismatch flag on an identical number.
        assert extract_numbers("Effects of AC2993.") == {"2993"}

    def test_same_number_matches_regardless_of_trailing_punctuation(self):
        with_period = extract_numbers("The dose was 2993.")
        without_period = extract_numbers("The dose was 2993 today")
        assert with_period == without_period


class TestVerifyCitations:
    def test_fabricated_citation_is_detected(self):
        evidence = [
            {"source_id": "NCT01057251", "source_type": "clinical_trial", "text": "Real evidence.", "metadata": {}}
        ]
        answer = "Fake finding [source: NCT99999999]."
        report = verify_citations(answer, evidence)
        assert report["fabricated_count"] == 1
        assert not report["passed"]

    def test_real_citation_with_matching_number_passes(self):
        evidence = [
            {
                "source_id": "NCT01057251",
                "source_type": "clinical_trial",
                "text": "The trial reported a 12 percent reduction.",
                "metadata": {},
            }
        ]
        answer = "The trial reported a 12% reduction [source: NCT01057251]."
        report = verify_citations(answer, evidence)
        assert report["numeric_mismatch_count"] == 0
        assert report["passed"]

    def test_unsupported_number_is_flagged(self):
        evidence = [
            {
                "source_id": "NCT01057251",
                "source_type": "clinical_trial",
                "text": "The trial reported a 12 percent reduction.",
                "metadata": {},
            }
        ]
        answer = "The trial reported a 23% reduction [source: NCT01057251]."
        report = verify_citations(answer, evidence)
        assert report["numeric_mismatch_count"] == 1
        assert not report["passed"]

    def test_multi_source_synthesis_does_not_false_positive(self):
        # Regression test: a real bug caused numbers legitimately belonging
        # to OTHER sources in a shared sentence to be flagged as
        # unsupported by each individual source. Grouping by shared
        # sentence and checking the UNION of evidence fixes this.
        evidence = [
            {"source_id": "NCT01057251", "source_type": "clinical_trial", "text": "Ages 18 to 64.", "metadata": {}},
            {"source_id": "NCT00728858", "source_type": "clinical_trial", "text": "Ages 20 to 90.", "metadata": {}},
        ]
        answer = "Ages ranged from 18-64 [source: NCT01057251] to 20-90 [source: NCT00728858]."
        report = verify_citations(answer, evidence)
        assert report["numeric_mismatch_count"] == 0

    def test_metadata_numbers_count_as_supported(self):
        evidence = [
            {
                "source_id": "NCT01057251",
                "source_type": "clinical_trial",
                "text": "This trial evaluated treatment efficacy.",
                "metadata": {"phases": ["PHASE2"]},
            }
        ]
        answer = "This was a Phase 2 trial [source: NCT01057251]."
        report = verify_citations(answer, evidence)
        assert report["numeric_mismatch_count"] == 0
