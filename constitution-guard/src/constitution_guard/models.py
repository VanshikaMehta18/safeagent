"""Core Pydantic models for Constitution Guard."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class IntentCategory(str, Enum):
    BENIGN = "benign"
    CRISIS = "crisis"
    HARMFUL = "harmful"
    JAILBREAK = "jailbreak"
    AMBIGUOUS = "ambiguous"


class PrincipleScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class SafetyScore(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0)
    principle_scores: dict[str, PrincipleScore | None]
    verdict: Verdict
    flagged_principles: list[str]
    reasoning_trace: list[str]
    scored_principles: list[str] = Field(default_factory=list)
    unscored_principles: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    reasoning: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardChecks(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)
    verdict: Verdict = Verdict.PASS

    @property
    def flagged_names(self) -> list[str]:
        return [r.name for r in self.results if r.flagged]


class GuardResult(BaseModel):
    content: str
    verdict: Verdict
    input_checks: GuardChecks
    output_checks: GuardChecks
    constitutional_score: SafetyScore | None = None
    was_rewritten: bool = False
    referral: str | None = None
    safety_note: str = ""
    agent_called: bool = True


class RewriteResponse(BaseModel):
    rewritten_draft: str
    adjustments: str
