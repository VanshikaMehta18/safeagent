"""Short-circuit handlers for harmful and crisis intents."""

from __future__ import annotations

import logging

from api.schemas import FinalResponse, IntentCategory, Verdict
from graph.state import AgentState
from observability.langsmith_tracer import traceable
from safety import logger as safety_logger

logger = logging.getLogger(__name__)

HARM_SUBTYPE_REFERRALS: dict[str, str] = {
    "malware": "a cybersecurity professional or your organization's IT security team",
    "weapons": "local law enforcement or a trusted authority figure",
    "overdose": "Poison Control (1-800-222-1222) or a licensed healthcare provider",
    "self_harm": "the 988 Suicide & Crisis Lifeline (call or text 988)",
    "fraud": "a qualified attorney or financial advisor",
    "violence": "local law enforcement or emergency services (911)",
    "jailbreak": "a qualified professional in the relevant field",
    "other": "a qualified professional in the relevant field",
}

CRISIS_MESSAGE = (
    "I'm concerned about what you're going through, and you don't have to face this alone. "
    "Please reach out for help — call or text 988 (Suicide & Crisis Lifeline), available 24/7. "
    "If you are in immediate danger, call 911 or go to your nearest emergency room. "
    "Talking to a counselor, trusted friend, or family member can make a real difference."
)

HARMFUL_BLOCK_TEMPLATE = (
    "I'm unable to help with this request because it involves {concern}. "
    "I recommend reaching out to {referral} for appropriate guidance."
)


def _referral_for_subtype(harm_subtype: str | None) -> str:
    if harm_subtype:
        return HARM_SUBTYPE_REFERRALS.get(harm_subtype, HARM_SUBTYPE_REFERRALS["other"])
    return HARM_SUBTYPE_REFERRALS["other"]


def _concern_label(harm_subtype: str | None, category: IntentCategory) -> str:
    labels = {
        "malware": "creating or distributing malicious software",
        "weapons": "weapons or explosives",
        "overdose": "lethal substance dosing",
        "self_harm": "self-harm",
        "fraud": "illegal or fraudulent activity",
        "violence": "violence or harm to others",
        "jailbreak": "bypassing safety protections",
    }
    if harm_subtype and harm_subtype in labels:
        return labels[harm_subtype]
    if category == IntentCategory.JAILBREAK:
        return "attempting to bypass safety protections"
    return "potentially harmful or illegal activity"


@traceable("safeagent.harmful_block")
async def run_harmful_block(state: AgentState) -> AgentState:
    """Short-circuit: return BLOCK response for harmful/jailbreak queries."""
    intent = state.get("intent_output")
    subtype = intent.harm_subtype if intent else None
    category = intent.category if intent else IntentCategory.HARMFUL

    referral = _referral_for_subtype(subtype)
    concern = _concern_label(subtype, category)
    content = HARMFUL_BLOCK_TEMPLATE.format(concern=concern, referral=referral)

    state["final_response"] = FinalResponse(
        content=content,
        safety_verdict=Verdict.BLOCK.value,
        was_rewritten=False,
        referral_suggestion=referral,
        safety_note=f"Blocked at intent layer: {intent.reasoning if intent else 'harmful intent detected'}",
    )

    await safety_logger.log_decision(
        query_id=state["query_id"],
        query=state["query"],
        query_type="intent_block",
        principle_scores={"intent_classifier": 0.0},
        overall_score=0.0,
        verdict=Verdict.BLOCK.value,
        flagged_principles=["intent_classifier"],
        was_rewritten=False,
        latency_ms=0.0,
    )
    logger.info("Harmful block handler: subtype=%s", subtype)
    return state


@traceable("safeagent.crisis_handler")
async def run_crisis_handler(state: AgentState) -> AgentState:
    """Short-circuit: return compassionate crisis response with WARN verdict."""
    intent = state.get("intent_output")
    reasoning = intent.reasoning if intent else "Crisis intent detected"

    state["final_response"] = FinalResponse(
        content=CRISIS_MESSAGE,
        safety_verdict=Verdict.WARN.value,
        was_rewritten=False,
        referral_suggestion="988 Suicide & Crisis Lifeline (call or text 988)",
        safety_note=f"Crisis response issued. Please reach out for help. ({reasoning})",
    )

    await safety_logger.log_decision(
        query_id=state["query_id"],
        query=state["query"],
        query_type="crisis",
        principle_scores={"intent_classifier": 0.5},
        overall_score=0.5,
        verdict=Verdict.WARN.value,
        flagged_principles=["crisis_intent"],
        was_rewritten=False,
        latency_ms=0.0,
    )
    logger.info("Crisis handler: issued compassionate redirect")
    return state
