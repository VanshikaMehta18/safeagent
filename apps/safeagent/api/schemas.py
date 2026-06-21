"""Pydantic models for SafeAgent API and agent outputs."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    GENERAL = "general"
    SENSITIVE = "sensitive"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class IntentCategory(str, Enum):
    BENIGN = "benign"
    EDUCATIONAL = "educational"
    CRISIS = "crisis"
    HARMFUL = "harmful"
    JAILBREAK = "jailbreak"
    AMBIGUOUS = "ambiguous"


class IntentOutput(BaseModel):
    """Structured output from the Intent Classifier (Agent 0)."""

    category: IntentCategory
    harm_subtype: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class IntentLLMResponse(BaseModel):
    """Raw LLM response for intent classification."""

    category: str
    harm_subtype: str | None = None
    confidence: float
    reasoning: str


class PlannerOutput(BaseModel):
    """Structured output from the Planner agent."""

    subtasks: list[str] = Field(..., min_length=2, max_length=4)
    query_type: QueryType
    requires_web_search: bool
    risk_level: RiskLevel


class Source(BaseModel):
    """A single research source with relevance score."""

    content: str
    url: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ResearchOutput(BaseModel):
    """Structured output from the Researcher agent."""

    sources: list[Source]


class SynthesizerOutput(BaseModel):
    """Structured output from the Synthesizer agent."""

    draft: str
    cited_sources: list[int]
    confidence: float = Field(..., ge=0.0, le=1.0)


class PrincipleScore(BaseModel):
    """Score for a single constitutional principle."""

    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class SafetyScore(BaseModel):
    """Aggregated safety arbitration result."""

    overall_score: float = Field(..., ge=0.0, le=1.0)
    principle_scores: dict[str, PrincipleScore | None]
    verdict: Verdict
    flagged_principles: list[str]
    reasoning_trace: list[str]
    scored_principles: list[str] = Field(default_factory=list)
    unscored_principles: list[str] = Field(default_factory=list)


class FinalResponse(BaseModel):
    """Final user-facing response after escalation."""

    content: str
    safety_verdict: str
    was_rewritten: bool
    referral_suggestion: str | None = None
    safety_note: str


class QueryRequest(BaseModel):
    """POST /query request body."""

    query: str = Field(..., min_length=1)
    user_id: str | None = None


class QueryResponse(BaseModel):
    """POST /query response."""

    query_id: str
    latency_ms: float
    response: FinalResponse


class SafetyLogEntry(BaseModel):
    """A single safety arbitration log entry."""

    timestamp: str
    query_id: str
    query_hash: str
    query_type: str
    principle_scores: dict[str, float]
    overall_score: float
    verdict: str
    flagged_principles: list[str]
    was_rewritten: bool
    latency_ms: float


class SafetyReport(BaseModel):
    """Aggregated safety statistics."""

    total_queries: int
    pass_count: int
    warn_count: int
    block_count: int
    most_flagged_principles: list[dict[str, int | str]]
    average_safety_score: float
    average_latency_ms: float


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    version: str


class SearchResult(BaseModel):
    """FAISS search result."""

    content: str
    metadata: dict
    score: float


class SeedDocumentsResponse(BaseModel):
    """Response model for LLM-generated seed documents."""

    documents: list[dict[str, str]]


class PrincipleEvalResponse(BaseModel):
    """Single principle evaluation from the LLM."""

    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class PlannerLLMResponse(BaseModel):
    """Raw structured response from planner LLM call."""

    subtasks: list[str]
    query_type: str
    requires_web_search: bool
    risk_level: str


class SynthesizerLLMResponse(BaseModel):
    """Raw structured response from synthesizer LLM call."""

    draft: str
    cited_sources: list[int]
    confidence: float


class RewriteLLMResponse(BaseModel):
    """Structured response from escalation rewrite."""

    rewritten_draft: str
    adjustments: str


class EvalPrompt(BaseModel):
    """Adversarial eval prompt definition."""

    prompt: str
    category: str
    expected_verdict: Literal["PASS", "WARN", "BLOCK"]
    notes: str


class EvalResult(BaseModel):
    """Result of a single eval prompt run."""

    prompt: str
    category: str
    expected_verdict: str
    actual_verdict: str
    overall_score: float
    latency_ms: float
    matched: bool


class EvalReport(BaseModel):
    """Full eval run report."""

    total: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    mean_latency_ms: float
    per_category: dict[str, dict[str, float]]
    results: list[EvalResult]
