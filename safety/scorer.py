"""Safety scoring and arbitration logic."""

from __future__ import annotations

import asyncio
import logging
import time

from api.schemas import IntentCategory, IntentOutput, PrincipleScore, SafetyScore, Verdict
from config import get_settings
from llm import call_llm_json
from safety.constitution import PRINCIPLES, ConstitutionalPrinciple

logger = logging.getLogger(__name__)

HARMFUL_SUBTYPES = frozenset({"malware", "weapons", "overdose", "self_harm", "violence", "fraud"})
INTENT_BLOCK_SCORE_CAP = 0.4
INTENT_CRISIS_SCORE_CAP = 0.6


class _PrincipleEvalModel(PrincipleScore):
    """Internal model for Gemini principle evaluation responses."""


async def score_principle(
    principle: ConstitutionalPrinciple,
    query: str,
    draft: str,
) -> PrincipleScore:
    """Score a single constitutional principle via Gemini critic prompt."""
    prompt = principle.critic_prompt_template.format(query=query, draft=draft)
    result = await call_llm_json(prompt, _PrincipleEvalModel)
    return PrincipleScore(score=result.data.score, reasoning=result.data.reasoning)


def _compute_verdict(overall_score: float) -> Verdict:
    settings = get_settings()
    if overall_score >= settings.safety_pass_threshold:
        return Verdict.PASS
    if overall_score >= settings.safety_warn_threshold:
        return Verdict.WARN
    return Verdict.BLOCK


def _apply_intent_overlay(
    overall_score: float,
    verdict: Verdict,
    intent_output: IntentOutput | None,
    reasoning_trace: list[str],
    flagged: list[str],
) -> tuple[float, Verdict, list[str]]:
    """Cap scores based on upstream intent classification."""
    if intent_output is None:
        return overall_score, verdict, flagged

    category = intent_output.category
    subtype = intent_output.harm_subtype

    if category == IntentCategory.CRISIS:
        if overall_score > INTENT_CRISIS_SCORE_CAP:
            overall_score = INTENT_CRISIS_SCORE_CAP
            reasoning_trace.append(
                f"intent_overlay: crisis intent capped score at {INTENT_CRISIS_SCORE_CAP}"
            )
            if "crisis_intent" not in flagged:
                flagged.append("crisis_intent")
        verdict = _compute_verdict(overall_score)
    elif category in (IntentCategory.HARMFUL, IntentCategory.JAILBREAK) or (
        subtype and subtype in HARMFUL_SUBTYPES
    ):
        if overall_score > INTENT_BLOCK_SCORE_CAP:
            overall_score = INTENT_BLOCK_SCORE_CAP
            reasoning_trace.append(
                f"intent_overlay: harmful intent ({category.value}/{subtype}) capped score at {INTENT_BLOCK_SCORE_CAP}"
            )
            if "intent_classifier" not in flagged:
                flagged.append("intent_classifier")
        verdict = _compute_verdict(overall_score)

    reasoning_trace.append(f"intent_classifier: {intent_output.reasoning}")
    return overall_score, verdict, flagged


async def arbitrate(
    query: str,
    draft: str,
    query_type: str,
    risk_level: str,
    intent_output: IntentOutput | None = None,
) -> tuple[SafetyScore, float]:
    """
    Run all constitutional principle critics and aggregate into a SafetyScore.

    Returns the SafetyScore and arbitration latency in milliseconds.
    """
    start = time.perf_counter()
    settings = get_settings()

    tasks = [score_principle(p, query, draft) for p in PRINCIPLES]
    scores = await asyncio.gather(*tasks)

    principle_scores: dict[str, PrincipleScore] = {}
    reasoning_trace: list[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for principle, score in zip(PRINCIPLES, scores):
        principle_scores[principle.name] = score
        reasoning_trace.append(f"{principle.name}: {score.reasoning}")
        weighted_sum += score.score * principle.weight
        weight_total += principle.weight

    overall_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    verdict = _compute_verdict(overall_score)

    flagged = [
        name
        for name, ps in principle_scores.items()
        if ps.score < settings.safety_warn_threshold
    ]

    if risk_level == "high" and verdict == Verdict.PASS and overall_score < 0.9:
        verdict = Verdict.WARN
        logger.info(
            "Elevated high-risk query %s to WARN despite score %.2f",
            query_type,
            overall_score,
        )

    overall_score, verdict, flagged = _apply_intent_overlay(
        overall_score, verdict, intent_output, reasoning_trace, flagged
    )

    latency_ms = (time.perf_counter() - start) * 1000

    safety_score = SafetyScore(
        overall_score=round(overall_score, 4),
        principle_scores=principle_scores,
        verdict=verdict,
        flagged_principles=flagged,
        reasoning_trace=reasoning_trace,
    )
    return safety_score, latency_ms
