"""
schemas.py

Pydantic models defining every request and response shape for the API.
FastAPI uses these automatically for:
  - Validating incoming requests (rejecting bad input with a clear 422 error
    before our code ever runs)
  - Generating the interactive API docs at /docs
  - Serializing responses consistently

Keeping these separate from main.py keeps the API's "data contract" easy to
read in one place, independent of the endpoint logic itself.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---------- Requests ----------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="The search query text")
    condition: Optional[str] = Field(None, description="Optional condition filter, e.g. 'hypertension'")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return (1-20)")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or only whitespace")
        return v.strip()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000, description="The question to answer")
    condition: Optional[str] = Field(None, description="Optional condition filter")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or only whitespace")
        return v.strip()


class CompareRequest(BaseModel):
    source_ids: list[str] = Field(..., min_length=2, max_length=5, description="2-5 NCT IDs or PMIDs to compare")
    aspect: Optional[str] = Field(None, max_length=200, description="What to compare, e.g. 'eligibility criteria'")

    @field_validator("source_ids")
    @classmethod
    def ids_not_blank(cls, v: list[str]) -> list[str]:
        cleaned = [sid.strip() for sid in v if sid.strip()]
        if len(cleaned) < 2:
            raise ValueError("at least 2 non-blank source_ids are required to compare")
        return cleaned


# ---------- Responses ----------


class ChunkResult(BaseModel):
    chunk_id: str
    source_type: str
    source_id: str
    title: str
    section_name: str
    text: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    condition_filter: Optional[str]
    results: list[ChunkResult]
    low_confidence: bool


class CitationInfo(BaseModel):
    source_type: str
    nct_id: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    section: str


class VerificationSummary(BaseModel):
    total_citations: int
    verified_count: int
    fabricated_count: int
    numeric_mismatch_count: int
    passed: bool


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    low_confidence: bool
    citations: list[CitationInfo]
    verification: Optional[VerificationSummary]


class CompareResponse(BaseModel):
    source_ids: list[str]
    aspect: Optional[str]
    comparison: str
    citations: list[CitationInfo]
    verification: Optional[VerificationSummary]


class SourceDetail(BaseModel):
    source_id: str
    source_type: str
    title: str
    sections: list[dict]
    metadata: dict


class HealthResponse(BaseModel):
    status: str
    corpus_loaded: bool
    vector_db_ready: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
