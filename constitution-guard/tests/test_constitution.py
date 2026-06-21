"""Tests for constitutional scoring."""

from __future__ import annotations

import pytest

from constitution_guard.config import GuardConfig
from constitution_guard.constitution.local_scorer import score_local_principles
from constitution_guard.constitution.principles import (
    LLM_PRINCIPLES,
    LOCAL_PRINCIPLES,
    PRINCIPLES,
)
from constitution_guard.constitution.scorer import compute_verdict
from constitution_guard.models import CheckResult, GuardChecks, PrincipleScore, Verdict


class MockCritic:
    def __init__(self) -> None:
        self.call_count = 0
        self.called_principles: list[str] = []

    async def score(self, principle, query, draft, context):
        self.call_count += 1
        self.called_principles.append(principle.name)
        return PrincipleScore(score=0.95, reasoning="safe")

    async def rewrite(self, query, draft, flagged, reasoning_trace):
        from constitution_guard.models import RewriteResponse
        return RewriteResponse(rewritten_draft=draft, adjustments="none")


def test_compute_verdict_thresholds():
    config = GuardConfig()
    assert compute_verdict(0.85, config) == Verdict.PASS
    assert compute_verdict(0.6, config) == Verdict.WARN
    assert compute_verdict(0.3, config) == Verdict.BLOCK


@pytest.mark.asyncio
async def test_arbitrate_hybrid_routing():
    from constitution_guard.constitution.scorer import arbitrate

    config = GuardConfig()
    critic = MockCritic()
    output_checks = GuardChecks(
        results=[
            CheckResult(name="toxicity", score=0.9, flagged=False, reasoning="Low toxicity"),
            CheckResult(name="pii", score=1.0, flagged=False, reasoning="No PII"),
            CheckResult(name="jailbreak", score=1.0, flagged=False, reasoning="Clean"),
        ]
    )
    score, latency = await arbitrate("test query", "safe draft", output_checks, critic, config)

    assert critic.call_count == len(LLM_PRINCIPLES)
    assert set(critic.called_principles) == {p.name for p in LLM_PRINCIPLES}
    assert len(score.principle_scores) == len(PRINCIPLES)
    assert score.unscored_principles == []
    assert len(score.scored_principles) == len(PRINCIPLES)
    assert all(ps is not None for ps in score.principle_scores.values())
    assert score.verdict == Verdict.PASS
    assert latency >= 0


@pytest.mark.asyncio
async def test_arbitrate_local_only():
    from constitution_guard.constitution.scorer import arbitrate

    config = GuardConfig()
    output_checks = GuardChecks(
        results=[
            CheckResult(name="toxicity", score=0.9, flagged=False, reasoning="Low toxicity"),
        ]
    )
    score, _ = await arbitrate("test query", "safe draft", output_checks, None, config)

    assert len(score.principle_scores) == len(PRINCIPLES)
    assert score.principle_scores["non_maleficence"] is not None
    assert score.principle_scores["non_maleficence"].score == 0.9
    assert score.principle_scores["honesty"] is None
    assert score.principle_scores["fairness"] is None
    assert score.principle_scores["autonomy_preservation"] is None
    assert set(score.unscored_principles) == {p.name for p in LLM_PRINCIPLES}
    assert len(score.scored_principles) == len(LOCAL_PRINCIPLES)
    assert any("not scored" in t for t in score.reasoning_trace)


@pytest.mark.asyncio
async def test_unscored_principles_excluded_from_verdict():
    """Low local scores must affect verdict even when LLM principles are unscored."""
    from constitution_guard.constitution.scorer import arbitrate

    config = GuardConfig()
    output_checks = GuardChecks(
        results=[
            CheckResult(
                name="toxicity",
                score=0.2,
                flagged=True,
                reasoning="Toxicity detected",
            ),
        ]
    )
    score, _ = await arbitrate("test query", "bad draft", output_checks, None, config)
    assert score.principle_scores["honesty"] is None
    assert score.verdict != Verdict.PASS
    assert score.flagged_principles == ["non_maleficence"]


def test_local_principle_mapping():
    output_checks = GuardChecks(
        results=[
            CheckResult(
                name="toxicity",
                score=0.2,
                flagged=True,
                reasoning="Toxicity detected (level=0.80)",
            ),
            CheckResult(
                name="jailbreak",
                score=0.1,
                flagged=True,
                reasoning="Prompt injection detected (risk=0.90)",
            ),
        ]
    )
    scores = score_local_principles(output_checks, "test", "draft")
    assert scores["non_maleficence"].score == 0.2
    assert scores["legality"].score == 0.1


def test_principles_count_and_routing():
    assert len(PRINCIPLES) == 7
    assert len(LOCAL_PRINCIPLES) == 4
    assert len(LLM_PRINCIPLES) == 3
    local_names = {p.name for p in LOCAL_PRINCIPLES}
    assert local_names == {
        "non_maleficence",
        "privacy",
        "legality",
        "scope_appropriateness",
    }
    llm_names = {p.name for p in LLM_PRINCIPLES}
    assert llm_names == {"honesty", "fairness", "autonomy_preservation"}
