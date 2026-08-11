"""
pubmed_client.py

Handles all communication with NCBI's E-utilities API for PubMed.
Two calls are needed to get abstracts:
  1. esearch  -> given a text query, returns a list of PMIDs (article IDs)
  2. efetch   -> given PMIDs, returns the actual article data (title, abstract, etc.)

We enforce NCBI's rate limit ourselves (10 requests/second with an API key,
3/second without) so we never get blocked for making requests too fast.
"""

import time
import xml.etree.ElementTree as ET

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import EUTILS_BASE_URL, NCBI_API_KEY, NCBI_EMAIL, NCBI_TOOL_NAME

# If we have an API key, NCBI allows 10 requests/second (1 every 0.1s).
# Without one, only 3/second (1 every 0.34s). We build in a small safety margin.
_MIN_DELAY_SECONDS = 0.11 if NCBI_API_KEY else 0.35
_last_request_time = 0.0


def _rate_limit():
    """Sleep just long enough to respect NCBI's rate limit before the next call."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_DELAY_SECONDS:
        time.sleep(_MIN_DELAY_SECONDS - elapsed)
    _last_request_time = time.time()


def _base_params() -> dict:
    """Common parameters NCBI asks every request to include."""
    params = {"tool": NCBI_TOOL_NAME, "email": NCBI_EMAIL}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_pubmed(query: str, max_results: int = 200, mindate: str = None, maxdate: str = None) -> list[str]:
    """
    Search PubMed for a text query and return a list of PMIDs (PubMed IDs).

    Note: esearch has a max retmax of 10000, far above what we need here.

    mindate / maxdate (optional, format "YYYY"): restrict results to a
    publication date range. This matters because PubMed's default ranking
    ("Best Match") gives recently-published articles a higher weight, so an
    unfiltered query mostly returns very recent papers even without
    explicitly sorting by date. To get a realistic historical spread across
    years, we call this function once per year range and combine the results
    (see fetch_pmids_across_years below).
    """
    _rate_limit()
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
    }
    if mindate and maxdate:
        params["mindate"] = mindate
        params["maxdate"] = maxdate
        params["datetype"] = "pdat"  # filter by publication date, not "date added"

    response = requests.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_pmids_across_years(query: str, total_results: int, years_back: int = 5) -> list[str]:
    """
    Fetch PMIDs spread evenly across the last `years_back` years, rather than
    relying on PubMed's default ranking (which skews heavily toward the most
    recent publications — see the docstring in search_pubmed).

    Splits total_results evenly across each year in the range and combines
    the results, so the final PMID list has genuine historical spread instead
    of being dominated by whichever year PubMed's algorithm favors.
    """
    from datetime import datetime

    current_year = datetime.now().year
    per_year = max(1, total_results // years_back)

    all_pmids = []
    for i in range(years_back):
        year = current_year - i
        pmids = search_pubmed(query, max_results=per_year, mindate=str(year), maxdate=str(year))
        all_pmids.extend(pmids)

    return all_pmids


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_abstracts(pmids: list[str]) -> str:
    """
    Given a list of PMIDs, fetch their full records (title, abstract, journal, etc.)
    as XML. efetch returns XML more reliably/completely than JSON for PubMed records,
    so we parse XML here rather than requesting JSON.
    """
    if not pmids:
        return ""

    _rate_limit()
    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    response = requests.get(f"{EUTILS_BASE_URL}/efetch.fcgi", params=params, timeout=30)
    response.raise_for_status()
    return response.text


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """
    Parse the raw XML from efetch into a list of flat dictionaries.

    PubMed's XML structure is deeply nested and inconsistent (e.g. abstracts
    can have multiple labeled sections like "BACKGROUND", "METHODS", "RESULTS").
    We join multi-part abstracts into one text field and keep it simple for now —
    section-aware parsing of abstracts happens later, in the chunking phase.
    """
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None

        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else None

        # Abstracts can be split into multiple <AbstractText> tags (e.g. one per section)
        abstract_parts = article_elem.findall(".//AbstractText")
        abstract = " ".join((part.text or "") for part in abstract_parts if part.text).strip()

        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Publication year extraction, with fallbacks.
        #
        # BUG FIX: the original code used ".//PubDate/Year", which searches
        # the ENTIRE article tree for any tag named PubDate. PubMed XML has
        # more than one: the real publication date lives under
        # Article/Journal/JournalIssue/PubDate, but there's also a PubDate
        # inside PubmedData/History (e.g. "date added to PubMed" / "date
        # revised") which is almost always very recent. ".//" matched
        # whichever one appeared first in document order, which was often
        # the history date — explaining why nearly every record showed the
        # current year regardless of when the paper was actually published.
        #
        # Fix: look specifically inside the Journal's PubDate first (the
        # true publication date). Some older/in-press records omit <Year>
        # and instead give a free-text <MedlineDate> like "2023 Jan-Feb" —
        # we fall back to extracting the leading 4-digit year from that.
        pub_year = None
        journal_pub_date = article_elem.find(".//Article/Journal/JournalIssue/PubDate")
        if journal_pub_date is not None:
            year_elem = journal_pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                pub_year = year_elem.text
            else:
                medline_date_elem = journal_pub_date.find("MedlineDate")
                if medline_date_elem is not None and medline_date_elem.text:
                    # e.g. "2023 Jan-Feb" -> take the first 4 digits
                    leading_digits = medline_date_elem.text.strip()[:4]
                    if leading_digits.isdigit():
                        pub_year = leading_digits

        # Last-resort fallback: electronic article date, if the journal
        # pub date was missing entirely.
        if pub_year is None:
            article_date_elem = article_elem.find(".//ArticleDate/Year")
            if article_date_elem is not None:
                pub_year = article_date_elem.text

        mesh_terms = [mesh.text for mesh in article_elem.findall(".//MeshHeading/DescriptorName") if mesh.text]

        # DOI and PMCID (PubMed Central ID) live in ArticleIdList, each
        # tagged with an IdType attribute. Not every article has both — a
        # PMCID only exists if the paper is deposited in PubMed Central, and
        # DOI is usually but not always present.
        #
        # BUG FIX: the original code searched ".//ArticleIdList/ArticleId",
        # which uses ".//" to search the ENTIRE article XML tree. PubMed
        # records include a <ReferenceList> listing every paper THIS article
        # cites, and each of those references has its OWN <ArticleIdList>
        # with its own DOI/PMCID. The unscoped search matched all of those
        # too, and since the code kept overwriting on each match, it ended
        # up storing whichever REFERENCE happened to appear last in the
        # document — a completely unrelated paper's identifiers, not the
        # actual article's own ones.
        #
        # Fix: scope the search to only the ArticleIdList that is a DIRECT
        # CHILD of PubmedData (the article's own ID list), excluding the
        # ones nested several levels deeper inside ReferenceList/Reference.
        doi = None
        pmcid = None
        for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
            id_type = article_id.get("IdType")
            if id_type == "doi":
                doi = article_id.text
            elif id_type == "pmc":
                pmcid = article_id.text

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "pub_year": pub_year,
                "mesh_terms": mesh_terms,
                "doi": doi,
                "pmcid": pmcid,
            }
        )

    return articles
