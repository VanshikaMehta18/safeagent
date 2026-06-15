"""Safety Arbitration agent — scores drafts against constitutional principles."""

from __future__ import annotations

import logging

from graph.state import AgentState
from observability.langsmith_tracer import traceable
from safety import logger as safety_logger
from safety.scorer import arbitrate

logger = logging.getLogger(__name__)


@traceable("safeagent.safety_arbitration")
async def run_safety_arbitration(state: AgentState) -> AgentState:
    """
    Score the synthesizer draft against all 7 constitutional principles.

    Core safety layer — every response must pass through here before reaching the user.
    Logs every decision to safety_log.jsonl.
    """
    if state.get("error") or not state.get("synthesizer_output"):
        return state

    query = state["query"]
    draft = state["synthesizer_output"].draft
    planner = state.get("planner_output")
    query_type = planner.query_type.value if planner else "general"
    risk_level = planner.risk_level.value if planner else "medium"

    try:
        safety_score, latency_ms = await arbitrate(
            query=query,
            draft=draft,
            query_type=query_type,
            risk_level=risk_level,
            intent_output=state.get("intent_output"),
        )
        state["safety_score"] = safety_score

        principle_scores_flat = {
            name: ps.score for name, ps in safety_score.principle_scores.items()
        }
        await safety_logger.log_decision(
            query_id=state["query_id"],
            query=query,
            query_type=query_type,
            principle_scores=principle_scores_flat,
            overall_score=safety_score.overall_score,
            verdict=safety_score.verdict.value,
            flagged_principles=safety_score.flagged_principles,
            was_rewritten=state.get("was_rewritten", False),
            latency_ms=latency_ms,
        )
        logger.info(
            "Safety arbitration: verdict=%s score=%.2f flagged=%s",
            safety_score.verdict.value,
            safety_score.overall_score,
            safety_score.flagged_principles,
        )
    except Exception as exc:
        logger.exception("Safety arbitration failed")
        state["error"] = f"Safety arbitration failed: {exc}"

    return state
