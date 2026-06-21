"""LangGraph pipeline state definition."""

from __future__ import annotations

from typing import TypedDict

from api.schemas import (
    FinalResponse,
    IntentOutput,
    PlannerOutput,
    ResearchOutput,
    SafetyScore,
    SynthesizerOutput,
)


class AgentState(TypedDict):
    """Shared state passed between all LangGraph nodes."""

    query: str
    query_id: str
    intent_output: IntentOutput | None
    planner_output: PlannerOutput | None
    research_output: ResearchOutput | None
    synthesizer_output: SynthesizerOutput | None
    safety_score: SafetyScore | None
    final_response: FinalResponse | None
    retry_count: int
    error: str | None
    needs_safety_recheck: bool
    block_on_escalation: bool
    was_rewritten: bool
    pipeline_start_ms: float
