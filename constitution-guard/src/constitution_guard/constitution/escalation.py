"""Escalation logic for constitutional verdicts."""

from __future__ import annotations

from constitution_guard.config import GuardConfig
from constitution_guard.models import SafetyScore, Verdict

REFERRAL_MAP = {
    "medical": "a licensed healthcare provider",
    "legal": "a qualified attorney",
    "financial": "a certified financial planner",
    "sensitive": "an appropriate counselor or specialist",
    "general": "a qualified professional in the relevant field",
}

BLOCK_REASONS = {
    "non_maleficence": "potential for harm",
    "legality": "attempted safety bypass or illegal facilitation",
    "jailbreak": "attempted safety bypass",
    "toxicity": "harmful or toxic content",
    "pii": "privacy concerns",
    "privacy": "privacy concerns",
    "crisis": "crisis-level concern",
    "fairness": "the response does not fairly address your request",
}


def _referral(query_type: str) -> str:
    return REFERRAL_MAP.get(query_type, REFERRAL_MAP["general"])


def _block_message(flagged: list[str], query_type: str) -> tuple[str, str]:
    top = flagged[0] if flagged else "scope_appropriateness"
    concern = BLOCK_REASONS.get(top, "safety concerns")
    referral = _referral(query_type)
    content = (
        f"I'm unable to provide this response because it raises {concern}. "
        f"Please consult {referral} for personalized guidance."
    )
    return content, referral


def apply_escalation(
    query: str,
    draft: str,
    safety_score: SafetyScore,
    config: GuardConfig,
    *,
    was_rewritten: bool = False,
    retry_count: int = 0,
) -> tuple[str, Verdict, str | None, str, bool]:
    """
    Returns (content, verdict, referral, safety_note, needs_rewrite).
    """
    if safety_score.verdict == Verdict.PASS:
        scored = len(safety_score.scored_principles)
        total = scored + len(safety_score.unscored_principles)
        if safety_score.unscored_principles:
            note = (
                f"Passed {scored}/{scored} scored constitutional checks "
                f"({len(safety_score.unscored_principles)} unscored: "
                f"{', '.join(safety_score.unscored_principles)})."
            )
        else:
            note = f"Passed {scored}/{total} constitutional checks."
        return draft, Verdict.PASS, None, note, False

    if safety_score.verdict == Verdict.BLOCK:
        content, referral = _block_message(safety_score.flagged_principles, config.query_type)
        return content, Verdict.BLOCK, referral, "Blocked by constitutional checks.", False

    if retry_count >= config.max_retries:
        content, referral = _block_message(safety_score.flagged_principles, config.query_type)
        content += " If you are in distress, please call or text 988."
        return (
            content,
            Verdict.BLOCK,
            referral,
            "Max retries exceeded. Please seek help.",
            False,
        )

    return draft, Verdict.WARN, None, "Rewrite required.", True
