"""Guard pipeline — input gate, agent, output gate, constitutional layer."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from constitution_guard.backends.critic import PrincipleCritic
from constitution_guard.backends.gemini import gemini_critic_from_config
from constitution_guard.classifiers.crisis import CRISIS_MESSAGE, detect_crisis
from constitution_guard.classifiers.jailbreak import JailbreakClassifier
from constitution_guard.classifiers.pii import PIIClassifier
from constitution_guard.classifiers.toxicity import ToxicityClassifier
from constitution_guard.config import GuardConfig
from constitution_guard.constitution.escalation import apply_escalation
from constitution_guard.constitution.scorer import arbitrate
from constitution_guard.models import CheckResult, GuardChecks, GuardResult, Verdict

logger = logging.getLogger(__name__)

AgentFn = Callable[..., str | Awaitable[str]]


def resolve_critic(
    config: GuardConfig, override: PrincipleCritic | None = None
) -> PrincipleCritic | None:
    """Resolve injected critic, config field, or deprecated gemini string."""
    if override is not None:
        return override
    if config.principle_critic is not None:
        return config.principle_critic
    if config.constitutional_backend == "gemini":
        if not config.gemini_api_key:
            logger.warning("constitutional_backend=gemini but GEMINI_API_KEY is empty")
            return None
        return gemini_critic_from_config(config)
    return None


def has_llm_critic(config: GuardConfig, critic: PrincipleCritic | None) -> bool:
    return critic is not None


def _build_classifiers(config: GuardConfig) -> list:
    classifiers = []
    if config.jailbreak:
        classifiers.append(JailbreakClassifier(config))
    if config.pii:
        classifiers.append(PIIClassifier(config))
    if config.toxicity:
        classifiers.append(ToxicityClassifier(config))
    return classifiers


def _run_classifiers(text: str, classifiers: list) -> GuardChecks:
    results: list[CheckResult] = []
    if not classifiers:
        return GuardChecks(results=[], verdict=Verdict.PASS)

    for clf in classifiers:
        try:
            results.append(clf.classify(text))
        except ImportError:
            logger.warning("Classifier %s skipped — install [classifiers] extra", clf.name)

    verdict = _aggregate_verdict(results)
    return GuardChecks(results=results, verdict=verdict)


def _aggregate_verdict(results: list[CheckResult]) -> Verdict:
    if any(r.name == "jailbreak" and r.flagged for r in results):
        return Verdict.BLOCK
    if any(r.name == "toxicity" and r.metadata.get("block") for r in results):
        return Verdict.BLOCK
    if any(r.name == "crisis" and r.flagged for r in results):
        return Verdict.WARN
    if any(r.flagged for r in results):
        return Verdict.WARN
    return Verdict.PASS


async def run_checks_async(text: str, classifiers: list, config: GuardConfig) -> GuardChecks:
    loop = asyncio.get_event_loop()
    results: list[CheckResult] = []
    if config.crisis:
        results.append(detect_crisis(text))
    if classifiers:
        for clf in classifiers:
            try:
                r = await loop.run_in_executor(None, clf.classify, text)
                results.append(r)
            except ImportError:
                logger.warning("Classifier %s skipped", clf.name)
    verdict = _aggregate_verdict(results)
    return GuardChecks(results=results, verdict=verdict)


def _block_result(
    content: str,
    verdict: Verdict,
    input_checks: GuardChecks,
    output_checks: GuardChecks,
    *,
    agent_called: bool = False,
    referral: str | None = None,
    note: str = "",
) -> GuardResult:
    return GuardResult(
        content=content,
        verdict=verdict,
        input_checks=input_checks,
        output_checks=output_checks,
        agent_called=agent_called,
        referral=referral,
        safety_note=note,
    )


async def _call_agent(fn: AgentFn, query: str) -> str:
    if inspect.iscoroutinefunction(fn):
        result = await fn(query)
    else:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, fn, query)
    return str(result)


async def run_guarded_pipeline(
    query: str,
    agent_fn: AgentFn,
    config: GuardConfig,
    critic: PrincipleCritic | None = None,
) -> GuardResult:
    critic = resolve_critic(config, critic)
    classifiers = _build_classifiers(config)

    input_checks = await run_checks_async(query, classifiers, config)

    if input_checks.verdict == Verdict.BLOCK:
        return _block_result(
            "I'm unable to process this request due to safety concerns.",
            Verdict.BLOCK,
            input_checks,
            GuardChecks(),
            agent_called=False,
            note="Blocked at input gate: " + ", ".join(input_checks.flagged_names),
        )

    if any(r.name == "crisis" and r.flagged for r in input_checks.results):
        return _block_result(
            CRISIS_MESSAGE,
            Verdict.WARN,
            input_checks,
            GuardChecks(),
            agent_called=False,
            referral="988 Suicide & Crisis Lifeline",
            note="Crisis detected — please reach out for help.",
        )

    draft = await _call_agent(agent_fn, query)
    output_checks = await run_checks_async(draft, classifiers, config)

    if output_checks.verdict == Verdict.BLOCK:
        return _block_result(
            "I'm unable to provide this response due to safety concerns in the generated content.",
            Verdict.BLOCK,
            input_checks,
            output_checks,
            agent_called=True,
            note="Blocked at output gate.",
        )

    constitutional_score, _ = await arbitrate(query, draft, output_checks, critic, config)
    content = draft
    verdict = constitutional_score.verdict
    referral = None
    note = "Passed local guardrails."
    was_rewritten = False

    if has_llm_critic(config, critic):
        retry = 0
        while retry <= config.max_retries:
            content, verdict, referral, note, needs_rewrite = apply_escalation(
                query, content, constitutional_score, config,
                was_rewritten=was_rewritten, retry_count=retry,
            )
            if not needs_rewrite:
                break
            rewrite = await critic.rewrite(  # type: ignore[union-attr]
                query, content,
                constitutional_score.flagged_principles,
                constitutional_score.reasoning_trace,
            )
            content = rewrite.rewritten_draft
            was_rewritten = True
            retry += 1
            note = f"Rewritten: {rewrite.adjustments}"
            output_checks = await run_checks_async(content, classifiers, config)
            constitutional_score, _ = await arbitrate(
                query, content, output_checks, critic, config
            )
    elif output_checks.verdict == Verdict.WARN:
        verdict = Verdict.WARN
        note = "WARN from local guardrails."

    return GuardResult(
        content=content,
        verdict=verdict,
        input_checks=input_checks,
        output_checks=output_checks,
        constitutional_score=constitutional_score,
        was_rewritten=was_rewritten,
        referral=referral,
        safety_note=note,
        agent_called=True,
    )
