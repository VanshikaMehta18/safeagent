"""Safety Arbitration agent — delegates to Constitution Guard."""

from __future__ import annotations

import logging

from graph.state import AgentState
from guard_client import get_guard
from observability.langsmith_tracer import traceable
from safety import logger as safety_logger

logger = logging.getLogger(__name__)


@traceable("safeagent.safety_arbitration")
async def run_safety_arbitration(state: AgentState) -> AgentState:
    """Score draft via Constitution Guard constitutional layer."""
    if state.get("error") or not state.get("synthesizer_output"):
        return state

    query = state["query"]
    draft = state["synthesizer_output"].draft
    planner = state.get("planner_output")
    query_type = planner.query_type.value if planner else "general"

    guard = get_guard()
    guard.config.query_type = query_type
    if planner:
        guard.config.risk_level = planner.risk_level.value

    try:
        result = await guard.arbitrate(query, draft)
        state["safety_score"] = result.constitutional_score

        if result.constitutional_score:
            score = result.constitutional_score
            principle_scores_flat = {
                name: ps.score
                for name, ps in score.principle_scores.items()
                if ps is not None
            }
            await safety_logger.log_decision(
                query_id=state["query_id"],
                query=query,
                query_type=query_type,
                principle_scores=principle_scores_flat,
                overall_score=score.overall_score,
                verdict=score.verdict.value,
                flagged_principles=score.flagged_principles,
                was_rewritten=state.get("was_rewritten", False),
                latency_ms=0.0,
            )
            logger.info(
                "Safety arbitration: verdict=%s score=%.2f",
                score.verdict.value,
                score.overall_score,
            )
    except Exception as exc:
        logger.exception("Safety arbitration failed")
        state["error"] = f"Safety arbitration failed: {exc}"

    return state
