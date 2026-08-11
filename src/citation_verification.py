"""
citation_verification.py

Fills Gap 2 from the Phase 9 diagram, and implements Phase 13's requirement
in full: CITATION VERIFICATION with an actual corrective action, not just a
report.

Three verification tiers run AFTER generation, on the model's raw output:

  TIER 1 - Fabrication check (hard check, always reliable):
    Extract every source ID the model cited (e.g. "NCT01057251"), and
    confirm that ID was actually present in the evidence we gave it. If the
    model cites a source ID that was never in the retrieved evidence, that's
    an unambiguous fabricated citation — a serious failure mode we can catch
    with 100% certainty, since we know exactly what evidence was provided.

  TIER 2 - Lexical support check (heuristic, not 100% reliable):
    For each citation, look at the sentence it's attached to, and check how
    much word overlap that sentence has with the actual text of the cited
    chunk. Low overlap suggests the claim might not really be supported by
    that source, even though the ID itself is real.

  TIER 3 - Numeric claim check (targeted, more precise than Tier 2 for
    numbers specifically):
    This is the exact scenario Phase 13 describes: "Trial X reported a 23%
    reduction" — does "23" actually appear anywhere in the cited evidence
    text? Numbers are the highest-stakes thing an LLM can hallucinate in a
    medical context (a wrong percentage is far more dangerous than a wrong
    adjective), so they get their own dedicated, stricter check rather than
    relying on Tier 2's whole-sentence word overlap, which could still pass
    a sentence containing a fabricated number as long as enough OTHER words
    in that sentence matched real evidence.

ACTION ON FAILURE (Phase 13's other requirement — previously missing):
Earlier, this module only REPORTED problems; nothing was done about them.
Now, verify_and_correct() in rag_pipeline.py acts on the result:
  - Fabricated citation or numeric mismatch -> attempt ONE regeneration,
    explicitly telling the model which claims failed and why
  - If the regenerated answer still fails -> refuse rather than present an
    unverified answer as if it were trustworthy

LIMITATIONS OF AUTOMATED CLAIM VERIFICATION (documented, not hidden):
  - Tier 2/3 are heuristics (word/number overlap), not true semantic
    entailment — a claim can be genuinely correct but phrased differently
    enough to score low (a real cost we saw in testing: "frequently
    excluded across multiple studies" scored 0.0 overlap despite being an
    accurate paraphrase, not a fabrication).
  - Tier 3 only checks whether a number appears in the text SOMEWHERE, not
    whether it modifies the same thing the claim says it does (e.g. a chunk
    mentioning "23%" for an unrelated statistic would pass, even if the
    claim misattributes it).
  - None of these tiers can catch subtle misrepresentation — e.g. citing a
    real, present number but describing its direction or context
    incorrectly (a decrease reported as an increase).
  - Regeneration is not guaranteed to fix a genuine problem — a model that
    hallucinated once can hallucinate again in a different way. The bounded
    single retry is a mitigation, not a guarantee.
  - A real production system would want a proper NLI (natural language
    inference) model for entailment checking, which is heavier to run but
    meaningfully more accurate than these lexical heuristics.
"""

import re


def extract_citations(answer_text: str) -> list[tuple[str, str]]:
    """
    Find every citation in the model's answer, e.g. "[source: NCT01057251]"
    or "[source: PMID 12345678]". Returns a list of (segment_text, source_id)
    pairs, where segment_text is the citation-bracket-free text of the
    SEGMENT (sentence/bullet) the citation belongs to.

    IMPORTANT: segment_text is computed by first splitting the whole answer
    into segments, THEN finding which segment each citation falls into —
    NOT by looking backward from each citation's own position. The earlier
    approach (looking backward per-citation) produced a DIFFERENT text
    string for every citation in a shared sentence (each one capturing
    progressively more text as more citations accumulate), which silently
    broke the grouping logic in verify_citations: citations that were
    genuinely part of the same sentence never matched as a group because
    their "sentence" strings were never actually equal. Segmenting the
    whole text once, up front, gives every citation in the same sentence
    the exact same segment_text, so grouping by it actually works.

    BUG FIX (found via real API testing — a serious one): the original
    citation pattern required a bracket to contain EXACTLY ONE id and close
    immediately after it: "[source: NCT01057251]". This assumed a specific
    formatting convention, but Gemini sometimes writes multiple sources in
    ONE bracket instead: "[source: NCT04434924, source: NCT03683069, source:
    NCT047251]". The old regex matched NOTHING in that case — not one of
    the IDs inside — meaning the ENTIRE group silently skipped verification,
    including a malformed ID that wasn't a real trial. This is exactly the
    failure mode citation verification exists to catch, and it was slipping
    through purely due to a formatting assumption, not because the ID was
    actually confirmed. Fix: find the full bracket SPAN first (from
    "[source:" to the next "]"), then extract every ID-like token from
    anywhere inside that span — this correctly handles a single ID, several
    IDs each with their own "source:" prefix, or a bare comma-separated list,
    without needing to know in advance which format the model used.
    """
    bracket_span_pattern = r"\[source:([^\]]*)\]"
    id_token_pattern = r"NCT\d+|PMID\s*\d+"

    # Split the whole answer into segments on real sentence/bullet boundaries
    segments = re.split(r"(?<=[.!?])\s+|\n[*\-]\s*", answer_text)

    citations = []
    for segment in segments:
        segment_clean = re.sub(bracket_span_pattern, "", segment).strip()
        for bracket_match in re.finditer(bracket_span_pattern, segment):
            bracket_content = bracket_match.group(1)
            for id_match in re.finditer(id_token_pattern, bracket_content, flags=re.IGNORECASE):
                raw_id = id_match.group(0)
                if raw_id.upper().startswith("NCT"):
                    source_id = raw_id.upper()
                else:
                    number = re.sub(r"[^\d]", "", raw_id)
                    source_id = f"PMID {number}"
                citations.append((segment_clean, source_id))

    return citations


def tokenize_words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def extract_numbers(text: str) -> set:
    """
    Extract numeric tokens (including percentages and decimals) from text,
    e.g. "23%" -> "23", "1.5 mg" -> "1.5". Used for Tier 3's targeted check
    on whether a cited number actually appears in the evidence.

    BUG FIX (found in real eval results): the original pattern r"\\d+\\.?\\d*"
    would match a trailing sentence-ending period as part of the number
    itself — e.g. "...effects of AC2993." at the end of a sentence extracted
    as "2993." (with the period), while the same drug name appearing
    mid-sentence elsewhere (e.g. in the model's answer) extracted as "2993"
    (no period). Since these are different strings, a genuinely identical
    number was incorrectly treated as unsupported. Fix: require an actual
    digit after the decimal point (\\d+(?:\\.\\d+)?), so a bare trailing
    period is never swept into the match.
    """
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def verify_citations(answer_text: str, evidence_chunks: list[dict], overlap_threshold: float = 0.15) -> dict:
    """
    Run all three verification tiers and return a structured report.

    evidence_chunks: the exact chunks that were given to the LLM, so we know
    the ground truth of what sources were actually available to cite.
    """
    evidence_by_source_id = {}
    for chunk in evidence_chunks:
        label = chunk["source_id"] if chunk["source_type"] == "clinical_trial" else f"PMID {chunk['source_id']}"
        evidence_by_source_id[label] = chunk

    citations = extract_citations(answer_text)

    fabricated = []
    weakly_supported = []
    numeric_mismatches = []
    verified = []

    # --- Group citations that share the exact same sentence ---
    # BUG FIX (found via real eval data — eval_017, a multi-document
    # synthesis question, produced 12 false-positive numeric mismatches):
    # a synthesis sentence often cites several sources together, each one
    # backing a DIFFERENT specific number, e.g. "Ages ranged from 18-64
    # [source: A], 20-90 [source: B], and 18-85 [source: C]". Checking each
    # citation's individual source against ALL numbers in the shared
    # sentence incorrectly flags B and C's numbers as unsupported by A,
    # even though every number is genuinely real — just attributed to a
    # different one of the cited sources. Fix: group citations by their
    # shared sentence, and check claimed numbers against the UNION of all
    # grouped sources' evidence, not each source individually.
    from collections import defaultdict

    sentence_groups = defaultdict(list)
    for sentence, source_id in citations:
        sentence_groups[sentence].append(source_id)

    for sentence, source_ids in sentence_groups.items():
        # Fabrication check still happens per-citation — a fabricated
        # source is fabricated regardless of what else it's grouped with.
        real_source_ids = []
        for source_id in source_ids:
            if source_id not in evidence_by_source_id:
                fabricated.append(source_id)
            else:
                real_source_ids.append(source_id)

        if not real_source_ids:
            continue

        # Numeric check against the UNION of all real sources in this group.
        #
        # BUG FIX (found in real eval results — eval_001, eval_010, eval_014,
        # eval_027, eval_029 all showed isolated numeric_mismatch=1 flags on
        # otherwise-correct answers): Phase 10 added trial METADATA (phase,
        # minimum age, status, etc.) into the prompt and explicitly told
        # Gemini it may use this metadata in its answers. But this numeric
        # check only ever looked at chunk["text"] — never chunk["metadata"].
        # So when the model correctly and honestly cited a number that only
        # appears in metadata (e.g. "Phase 2" from the phases field, or an
        # age boundary from minimum_age), it got wrongly flagged as
        # unsupported, since that number was never in the raw evidence text.
        # Fix: also extract numbers from a stringified version of the
        # chunk's metadata, so legitimately metadata-sourced numbers count
        # as supported.
        claimed_numbers = extract_numbers(sentence)
        union_evidence_numbers = set()
        for source_id in real_source_ids:
            chunk = evidence_by_source_id[source_id]
            union_evidence_numbers |= extract_numbers(chunk["text"])
            union_evidence_numbers |= extract_numbers(str(chunk.get("metadata", {})))
        unsupported_numbers = claimed_numbers - union_evidence_numbers

        if unsupported_numbers:
            for source_id in real_source_ids:
                numeric_mismatches.append(
                    {
                        "source_id": source_id,
                        "sentence": sentence[:150],
                        "unsupported_numbers": list(unsupported_numbers),
                    }
                )
            continue  # skip Tier 2 for this group — numeric mismatch is the more serious flag

        # Tier 2: lexical overlap, still checked per-citation (a sentence's
        # words should reasonably overlap with EACH of its cited sources,
        # even if numbers are split across them)
        for source_id in real_source_ids:
            chunk = evidence_by_source_id[source_id]
            sentence_words = tokenize_words(sentence)
            chunk_words = tokenize_words(chunk["text"])

            if not sentence_words:
                continue

            overlap = len(sentence_words & chunk_words) / len(sentence_words)

            if overlap < overlap_threshold:
                weakly_supported.append(
                    {
                        "source_id": source_id,
                        "sentence": sentence[:150],
                        "overlap_score": round(overlap, 3),
                    }
                )
            else:
                verified.append(
                    {
                        "source_id": source_id,
                        "overlap_score": round(overlap, 3),
                    }
                )

    total_citations = len(citations)
    return {
        "total_citations": total_citations,
        "verified_count": len(verified),
        "weakly_supported_count": len(weakly_supported),
        "fabricated_count": len(fabricated),
        "numeric_mismatch_count": len(numeric_mismatches),
        "fabricated_source_ids": fabricated,
        "weakly_supported_details": weakly_supported,
        "numeric_mismatch_details": numeric_mismatches,
        "verified_details": verified,
        # A hard failure = either a fabricated source OR a claimed number
        # that doesn't appear anywhere in its cited evidence. Both are
        # serious enough to trigger the regenerate/refuse action.
        "passed": len(fabricated) == 0 and len(numeric_mismatches) == 0,
    }


def print_verification_report(report: dict):
    print("\n" + "=" * 60)
    print("CITATION VERIFICATION")
    print("=" * 60)
    print(f"Total citations found: {report['total_citations']}")
    print(f"  Verified (real source, good lexical overlap): {report['verified_count']}")
    print(f"  Weakly supported (real source, low overlap): {report['weakly_supported_count']}")
    print(f"  NUMERIC MISMATCH (cited number not found in evidence): {report['numeric_mismatch_count']}")
    print(f"  FABRICATED (source not in evidence given): {report['fabricated_count']}")

    if report["fabricated_source_ids"]:
        print(f"\n  ⚠️  FABRICATED CITATIONS DETECTED: {report['fabricated_source_ids']}")
        print("  The model cited a source that was never provided to it — this is a hard failure.")

    if report["numeric_mismatch_details"]:
        print("\n  ⚠️  NUMERIC MISMATCHES DETECTED (hard failure):")
        for detail in report["numeric_mismatch_details"]:
            print(
                f'    - [{detail["source_id"]}] numbers {detail["unsupported_numbers"]} not found in evidence: "{detail["sentence"]}"'
            )

    if report["weakly_supported_details"]:
        print("\n  Weakly supported claims (real source, but low word overlap — review manually):")
        for detail in report["weakly_supported_details"]:
            print(f'    - [{detail["source_id"]}] overlap={detail["overlap_score"]}: "{detail["sentence"]}"')

    status = "PASSED" if report["passed"] else "FAILED (fabricated citation or unsupported number found)"
    print(f"\nVerification status: {status}")
