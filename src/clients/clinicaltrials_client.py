"""
clinicaltrials_client.py

Handles all communication with the ClinicalTrials.gov API v2.
No API key is required for this API, but we still add retry logic because
network calls fail sometimes (timeouts, temporary server errors) and we
don't want the whole pipeline to crash on one bad request.
"""

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import CLINICALTRIALS_BASE_URL


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _get(params: dict) -> dict:
    """
    Low-level GET request with automatic retry.

    Retries up to 3 times with increasing wait (2s, 4s, 8s) if the request
    fails — this handles transient network issues without manual intervention.
    """
    response = requests.get(CLINICALTRIALS_BASE_URL, params=params, timeout=30)
    response.raise_for_status()  # raises an exception on 4xx/5xx responses
    return response.json()


def fetch_trials_for_condition(condition: str, max_records: int = 200) -> list[dict]:
    """
    Fetch up to `max_records` completed interventional trials for a given condition.

    The API paginates results using a `nextPageToken`. We loop until we have
    enough records or the API tells us there are no more pages.

    Returns a list of raw study dicts (unmodified API response objects) —
    normalization into our own schema happens in a separate step, so we
    always keep the original raw data too (see ingest_trials.py).
    """
    all_studies = []
    page_token = None
    page_size = min(100, max_records)  # API max page size is 100

    while len(all_studies) < max_records:
        params = {
            "query.cond": condition,
            "filter.overallStatus": "COMPLETED",
            "pageSize": page_size,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        data = _get(params)
        studies = data.get("studies", [])
        all_studies.extend(studies)

        page_token = data.get("nextPageToken")
        if not page_token or not studies:
            break  # no more pages available

    return all_studies[:max_records]


def normalize_trial(raw_study: dict) -> dict:
    """
    Extract the fields we actually need from the deeply nested raw API response
    into a flat, predictable dictionary. This is the "translation layer" between
    ClinicalTrials.gov's schema and our own project's schema.

    We use .get() with defaults everywhere because, as noted in Phase 1 research,
    many fields are genuinely optional and missing in real records.
    """
    protocol = raw_study.get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    sponsor = protocol.get("sponsorCollaboratorsModule", {})
    description = protocol.get("descriptionModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    design = protocol.get("designModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    outcomes = protocol.get("outcomesModule", {})

    return {
        "nct_id": identification.get("nctId"),
        "brief_title": identification.get("briefTitle"),
        "official_title": identification.get("officialTitle"),
        "overall_status": status.get("overallStatus"),
        "start_date": status.get("startDateStruct", {}).get("date"),
        "completion_date": status.get("completionDateStruct", {}).get("date"),
        "lead_sponsor": sponsor.get("leadSponsor", {}).get("name"),
        "conditions": conditions_module.get("conditions", []),
        "study_type": design.get("studyType"),
        "phases": design.get("phases", []),
        "eligibility_criteria_text": eligibility.get("eligibilityCriteria"),
        "minimum_age": eligibility.get("minimumAge"),
        "sex": eligibility.get("sex"),
        "brief_summary": description.get("briefSummary"),
        "detailed_description": description.get("detailedDescription"),
        "primary_outcomes": outcomes.get("primaryOutcomes", []),
        "secondary_outcomes": outcomes.get("secondaryOutcomes", []),
    }
