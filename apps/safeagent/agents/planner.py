"""Planner agent — decomposes user queries into subtasks."""

from __future__ import annotations

import logging

from api.schemas import PlannerLLMResponse, PlannerOutput, QueryType, RiskLevel
from graph.state import AgentState
from llm import call_llm_json
from observability.langsmith_tracer import traceable

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = (
    "You are a query planning agent. Decompose user queries into actionable subtasks "
    "and classify them for a safety-aware research pipeline."
)


@traceable("safeagent.planner")
async def run_planner(state: AgentState) -> AgentState:
    """
    Decompose the user query into 2-4 subtasks and classify query type and risk.

    First node in the safety pipeline; sets initial risk estimate before arbitration.
    """
    query = state["query"]
    prompt = f"""Analyze this user query and create a research plan.

User query: {query}

Provide:
- subtasks: 2-4 specific research subtasks (list of strings)
- query_type: one of "medical", "legal", "financial", "general", "sensitive"
- requires_web_search: true if current/recent information is needed
- risk_level: initial safety estimate — "low", "medium", or "high"
"""

    try:
        result = await call_llm_json(
            prompt, PlannerLLMResponse, system=PLANNER_SYSTEM
        )
        raw = result.data
        subtasks = raw.subtasks[:4]
        if len(subtasks) < 2:
            subtasks = [query, f"Research context for: {query}"]

        try:
            query_type = QueryType(raw.query_type)
        except ValueError:
            query_type = QueryType.GENERAL
        try:
            risk_level = RiskLevel(raw.risk_level)
        except ValueError:
            risk_level = RiskLevel.MEDIUM

        planner_output = PlannerOutput(
            subtasks=subtasks,
            query_type=query_type,
            requires_web_search=raw.requires_web_search,
            risk_level=risk_level,
        )
        state["planner_output"] = planner_output
        logger.info(
            "Planner: type=%s risk=%s subtasks=%d",
            planner_output.query_type,
            planner_output.risk_level,
            len(planner_output.subtasks),
        )
    except Exception as exc:
        logger.exception("Planner agent failed")
        state["error"] = f"Planner failed: {exc}"

    return state
