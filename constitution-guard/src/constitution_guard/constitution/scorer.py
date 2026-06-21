"""Constitutional scoring — hybrid local + LLM routing."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from constitution_guard.backends.critic import PrincipleCritic
from constitution_guard.config import GuardConfig
from constitution_guard.constitution.local_scorer import score_local_principles
from constitution_guard.constitution.principles import LLM_PRINCIPLES, LOCAL_PRINCIPLES, PRINCIPLES
from constitution_guard.models import GuardChecks, PrincipleScore, SafetyScore, Verdict

logger = logging.getLogger(__name__)

_UNSCORED_REASON = "not scored (no principle critic configured)"


def compute_verdict(overall_score: float, config: GuardConfig) -> Verdict:
    if overall_score >= config.pass_threshold:
        return Verdict.PASS
    if overall_score >= config.warn_threshold:
        return Verdict.WARN
    return Verdict.BLOCK


def _build_safety_score(
    local_scores: dict[str, PrincipleScore],
    llm_scores: dict[str, PrincipleScore] | None,
    config: GuardConfig,
) -> SafetyScore:
    principle_scores: dict[str, PrincipleScore | None] = {}
    scored_principles: list[str] = []
    unscored_principles: list[str] = []
    reasoning_trace: list[str] = []

    for principle in LOCAL_PRINCIPLES:
        ps = local_scores.get(principle.name)
        principle_scores[principle.name] = ps
        if ps is not None:
            scored_principles.append(principle.name)
            reasoning_trace.append(f"{principle.name}: {ps.reasoning}")

    for principle in LLM_PRINCIPLES:
        ps = llm_scores.get(principle.name) if llm_scores else None
        principle_scores[principle.name] = ps
        if ps is not None:
            scored_principles.append(principle.name)
            reasoning_trace.append(f"{principle.name}: {ps.reasoning}")
        else:
            unscored_principles.append(principle.name)
            reasoning_trace.append(f"{principle.name}: {_UNSCORED_REASON}")

    weighted_sum = 0.0
    weight_total = 0.0
    for principle in PRINCIPLES:
        ps = principle_scores.get(principle.name)
        if ps is None:
            continue
        weighted_sum += ps.score * principle.weight
        weight_total += principle.weight

    overall = weighted_sum / weight_total if weight_total else 0.0
    verdict = compute_verdict(overall, config)
    flagged = [
        name
        for name, ps in principle_scores.items()
        if ps is not None and ps.score < config.warn_threshold
    ]

    if config.risk_level == "high" and verdict == Verdict.PASS and overall < 0.9:
        verdict = Verdict.WARN

    if unscored_principles:
        logger.info(
            "Constitutional score computed from %d/%d principles; unscored: %s",
            len(scored_principles),
            len(PRINCIPLES),
            ", ".join(unscored_principles),
        )

    return SafetyScore(
        overall_score=round(overall, 4),
        principle_scores=principle_scores,
        verdict=verdict,
        flagged_principles=flagged,
        reasoning_trace=reasoning_trace,
        scored_principles=scored_principles,
        unscored_principles=unscored_principles,
    )


async def arbitrate(
    query: str,
    draft: str,
    output_checks: GuardChecks,
    critic: PrincipleCritic | None,
    config: GuardConfig,
) -> tuple[SafetyScore, float]:
    start = time.perf_counter()

    local_scores = score_local_principles(output_checks, query, draft)
    context: dict[str, Any] = {"output_checks": output_checks}

    llm_scores: dict[str, PrincipleScore] | None = None
    if critic is not None:
        llm_tasks = [critic.score(p, query, draft, context) for p in LLM_PRINCIPLES]
        llm_results = await asyncio.gather(*llm_tasks)
        llm_scores = {p.name: score for p, score in zip(LLM_PRINCIPLES, llm_results)}

    latency = (time.perf_counter() - start) * 1000
    return _build_safety_score(local_scores, llm_scores, config), latency
