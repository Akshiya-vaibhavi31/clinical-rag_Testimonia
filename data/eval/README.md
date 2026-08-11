# Phase 14 Evaluation Set — How This Was Built

## Honest disclosure: how this dataset was constructed

I do not have live access to your actual vector database or corpus files. This
evaluation set was built using **real trial/paper data you pasted into our
conversation** during Phases 6-13 testing — actual NCT IDs, PMIDs, and actual
text excerpts that appeared in your terminal output when you ran real queries
against your real corpus.

This means:
- Every fact referenced (ages, percentages, GFR thresholds, etc.) is genuinely
  real data from your corpus, not invented — I did not make up any trial
  results or statistics.
- However, I have NOT independently re-verified every single entry against
  your live corpus myself (I can't — no direct access). You should spot-check
  a sample before treating this as fully "expert-reviewed" in the sense Phase
  14 asks for.

## What to do before trusting this fully

Pick 5-10 questions at random from `eval_set.json` and run them through
`inspect_citation.py` to confirm the `expected_source_ids` and
`relevant_evidence_snippet` fields are accurate:

```bash
python -m src.inspect_citation --id NCT01057251
```

If anything doesn't match what you see, that's expected — this is a first
draft built from partial visibility into your corpus during testing, not a
full independent audit. Fix any mismatches directly in `eval_set.json` (it's
plain JSON, editable in any text editor).

## Structure

32 questions across all 10 required categories:

| Category | Count |
|---|---|
| fact_lookup | 4 |
| eligibility | 4 |
| trial_comparison | 2 |
| outcome_extraction | 4 |
| adverse_events | 2 |
| study_design | 3 |
| intervention | 2 |
| multi_document_synthesis | 3 |
| unanswerable | 4 |
| adversarial | 4 |

23 answerable, 9 expected-refusal (unanswerable + adversarial combined) — a
deliberate mix so the evaluation actually tests refusal behavior, not just
"can it find the right chunk."

## Field meanings

- `expected_answerable`: should the system be ABLE to answer this from the corpus?
- `expected_source_ids`: which NCT ID(s)/PMID(s) should the answer cite, if answerable
- `relevant_evidence_snippet`: a short quote confirming the fact exists in that source
- `expected_answer_summary`: the correct answer in brief, for comparing against generated output
- `expected_refusal_reason`: WHY this should be refused, for unanswerable/adversarial questions

## Adversarial question design note

The 4 adversarial questions specifically test different attack patterns:
- `eval_022`: direct prompt injection ("ignore previous instructions")
- `eval_023`: appeal-to-authority pressure to confirm a fabricated statistic
- `eval_024`: leading question presupposing a false premise
- `eval_031`: requesting subjective opinion outside the system's evidence-based role

These are the kinds of adversarial patterns a real evaluation should include —
not just "hard questions," but questions designed to probe whether the safety
rules hold up under social engineering, not just under neutral phrasing.

## Next step

This dataset is input to Phase 15 (RAG Evaluation) — a script that runs every
question through your full pipeline, checks whether the actual behavior
matched `expected_answerable`, whether citations matched `expected_source_ids`,
and computes real metrics (refusal precision/recall, citation accuracy) across
all 32 questions automatically.
