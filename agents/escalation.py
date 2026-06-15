"""Escalation agent — handles PASS/WARN/BLOCK verdicts and produces final response."""

from __future__ import annotations

import logging

from api.schemas import FinalResponse, RewriteLLMResponse, Verdict
from config import get_settings
from graph.state import AgentState
from llm import call_llm_json
from observability.langsmith_tracer import traceable
from safety.constitution import PRINCIPLE_COUNT

logger = logging.getLogger(__name__)

REFERRAL_MAP: dict[str, str] = {
    "medical": "a licensed healthcare provider or your primary care physician",
    "legal": "a qualified attorney licensed in your jurisdiction",
    "financial": "a certified financial planner or licensed financial advisor",
    "sensitive": "an appropriate professional counselor or specialist",
    "general": "a qualified professional in the relevant field",
}

BLOCK_REASON_MAP: dict[str, str] = {
    "non_maleficence": "potential for physical, psychological, or financial harm",
    "honesty": "concerns about accuracy or truthfulness of the information",
    "autonomy_preservation": "risk of manipulating or undermining your decision-making",
    "privacy": "risk of exposing or mishandling private information",
    "scope_appropriateness": "this topic requires expertise beyond what an AI assistant can safely provide",
    "reversibility": "potential for irreversible consequences if acted upon",
    "consent_alignment": "the response does not appropriately align with your request",
}

REWRITE_SYSTEM = (
    "You are a safety editor. Rewrite AI responses to address flagged safety concerns "
    "while preserving helpful, accurate information."
)


def _get_referral(query_type: str) -> str:
    return REFERRAL_MAP.get(query_type, REFERRAL_MAP["general"])


def _build_block_message(
    query_type: str,
    flagged_principles: list[str],
) -> tuple[str, str]:
    top_flag = flagged_principles[0] if flagged_principles else "scope_appropriateness"
    concern = BLOCK_REASON_MAP.get(top_flag, "safety concerns")
    referral = _get_referral(query_type)

    content = (
        f"I'm unable to provide this response because it raises {concern}. "
        f"For questions in this area, I recommend consulting {referral} "
        f"who can give you personalized, reliable guidance."
    )
    return content, referral


def _build_pass_note() -> str:
    return f"This response passed {PRINCIPLE_COUNT}/{PRINCIPLE_COUNT} constitutional safety checks."


@traceable("safeagent.escalation")
async def run_escalation(state: AgentState) -> AgentState:
    """
    Produce the final user-facing response based on safety verdict.

    PASS: return draft with safety note.
    WARN: rewrite draft to fix flagged principles, trigger re-check.
    BLOCK: return constructive refusal with referral suggestion.
    """
    if state.get("error"):
        return state

    settings = get_settings()
    safety_score = state.get("safety_score")
    synthesizer = state.get("synthesizer_output")
    planner = state.get("planner_output")

    if not safety_score or not synthesizer:
        state["error"] = "Missing safety score or synthesizer output"
        return state

    query_type = planner.query_type.value if planner else "general"
    verdict = safety_score.verdict
    draft = synthesizer.draft

    try:
        if state.get("block_on_escalation"):
            content, referral = _build_block_message(
                query_type, safety_score.flagged_principles
            )
            state["final_response"] = FinalResponse(
                content=content,
                safety_verdict=Verdict.BLOCK.value,
                was_rewritten=state.get("was_rewritten", False),
                referral_suggestion=referral,
                safety_note="Maximum safety retries exceeded.",
            )
            return state

        if verdict == Verdict.PASS:
            state["final_response"] = FinalResponse(
                content=draft,
                safety_verdict=Verdict.PASS.value,
                was_rewritten=state.get("was_rewritten", False),
                referral_suggestion=None,
                safety_note=_build_pass_note(),
            )
            state["needs_safety_recheck"] = False
            return state

        if verdict == Verdict.BLOCK:
            content, referral = _build_block_message(
                query_type, safety_score.flagged_principles
            )
            state["final_response"] = FinalResponse(
                content=content,
                safety_verdict=Verdict.BLOCK.value,
                was_rewritten=False,
                referral_suggestion=referral,
                safety_note="Response blocked by constitutional safety checks.",
            )
            state["needs_safety_recheck"] = False
            return state

        # WARN — rewrite and re-check (retry_count incremented here, not in router)
        if state["retry_count"] >= settings.max_safety_retries:
            state["block_on_escalation"] = True
            content, referral = _build_block_message(
                query_type, safety_score.flagged_principles
            )
            content = (
                f"{content} If you are in distress, please reach out for help — "
                f"call or text 988 (Suicide & Crisis Lifeline) or contact {referral}."
            )
            state["final_response"] = FinalResponse(
                content=content,
                safety_verdict=Verdict.BLOCK.value,
                was_rewritten=state.get("was_rewritten", False),
                referral_suggestion=referral,
                safety_note="Unable to produce a safe response after multiple revisions. Please seek help.",
            )
            state["needs_safety_recheck"] = False
            return state

        state["retry_count"] = state.get("retry_count", 0) + 1
        flagged = ", ".join(safety_score.flagged_principles)
        rewrite_prompt = f"""Rewrite this AI draft to address the following safety concerns: {flagged}

Original query: {state['query']}
Original draft: {draft}

Flagged principles reasoning:
{chr(10).join(safety_score.reasoning_trace)}

Return JSON with:
- rewritten_draft: the improved response
- adjustments: brief description of what was changed and why
"""
        result = await call_llm_json(
            rewrite_prompt, RewriteLLMResponse, system=REWRITE_SYSTEM
        )
        state["synthesizer_output"] = synthesizer.model_copy(
            update={"draft": result.data.rewritten_draft}
        )
        state["was_rewritten"] = True
        state["needs_safety_recheck"] = True
        state["final_response"] = None

        logger.info(
            "Escalation: rewrite attempt %d/%d for principles: %s",
            state["retry_count"],
            settings.max_safety_retries,
            flagged,
        )

    except Exception as exc:
        logger.exception("Escalation agent failed")
        state["error"] = f"Escalation failed: {exc}"

    return state
