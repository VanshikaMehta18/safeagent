"""Intent Classifier (Agent 0) — uses Constitution Guard local classifiers."""

from __future__ import annotations

import logging

from api.schemas import IntentCategory, IntentOutput
from constitution_guard.models import Verdict
from graph.state import AgentState
from guard_client import get_guard
from observability.langsmith_tracer import traceable

logger = logging.getLogger(__name__)


def _checks_to_intent(checks) -> IntentOutput:
    flagged = {r.name: r for r in checks.results if r.flagged}
    if "crisis" in flagged:
        return IntentOutput(
            category=IntentCategory.CRISIS,
            harm_subtype="self_harm",
            confidence=0.9,
            reasoning=flagged["crisis"].reasoning,
        )
    if "jailbreak" in flagged:
        return IntentOutput(
            category=IntentCategory.JAILBREAK,
            harm_subtype="jailbreak",
            confidence=0.95,
            reasoning=flagged["jailbreak"].reasoning,
        )
    if checks.verdict == Verdict.BLOCK:
        return IntentOutput(
            category=IntentCategory.HARMFUL,
            harm_subtype="other",
            confidence=0.85,
            reasoning="Blocked by local classifiers: " + ", ".join(checks.flagged_names),
        )
    if checks.verdict == Verdict.WARN:
        return IntentOutput(
            category=IntentCategory.AMBIGUOUS,
            harm_subtype=None,
            confidence=0.6,
            reasoning="WARN from local classifiers.",
        )
    return IntentOutput(
        category=IntentCategory.BENIGN,
        harm_subtype=None,
        confidence=0.95,
        reasoning="Passed local classifier input gate.",
    )


@traceable("safeagent.intent_classifier")
async def run_intent_classifier(state: AgentState) -> AgentState:
    """Classify query intent using Constitution Guard local classifiers."""
    guard = get_guard()
    try:
        checks = await guard.check_input(state["query"])
        state["intent_output"] = _checks_to_intent(checks)
        logger.info(
            "Intent (via Guard): category=%s verdict=%s",
            state["intent_output"].category,
            checks.verdict,
        )
    except Exception as exc:
        logger.exception("Intent classifier failed")
        state["error"] = f"Intent classifier failed: {exc}"
        state["intent_output"] = IntentOutput(
            category=IntentCategory.AMBIGUOUS,
            harm_subtype=None,
            confidence=0.0,
            reasoning=str(exc),
        )
    return state
