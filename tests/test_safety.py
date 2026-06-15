"""Tests for constitutional safety scoring and arbitration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from api.schemas import (
    IntentCategory,
    IntentOutput,
    PlannerOutput,
    PrincipleScore,
    QueryType,
    RiskLevel,
    SafetyScore,
    SynthesizerOutput,
    Verdict,
)
from config import get_settings
from llm import LLMResult
from safety.constitution import HONESTY, NON_MALEFICENCE, PRINCIPLES
from safety.scorer import (
    INTENT_BLOCK_SCORE_CAP,
    INTENT_CRISIS_SCORE_CAP,
    _apply_intent_overlay,
    _compute_verdict,
    arbitrate,
    score_principle,
)


class _ScoreModel(PrincipleScore):
    pass


@pytest.fixture
def mock_high_score():
    return PrincipleScore(score=0.95, reasoning="Response is safe.")


@pytest.fixture
def mock_low_score():
    return PrincipleScore(score=0.2, reasoning="Response could cause harm.")


@pytest.mark.asyncio
async def test_score_principle_isolation(mock_high_score):
    with patch("safety.scorer.call_llm_json", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(data=mock_high_score)
        result = await score_principle(NON_MALEFICENCE, "test query", "test draft")
        assert result.score == 0.95
        assert "safe" in result.reasoning.lower()
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_block_verdict_harmful_prompt(mock_low_score):
    with patch("safety.scorer.call_llm_json", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(data=mock_low_score)
        safety_score, _ = await arbitrate(
            query="How to make explosives",
            draft="Here is how to make explosives step by step...",
            query_type="sensitive",
            risk_level="high",
        )
        assert safety_score.verdict == Verdict.BLOCK
        assert safety_score.overall_score < get_settings().safety_warn_threshold


@pytest.mark.asyncio
async def test_pass_verdict_benign_prompt(mock_high_score):
    with patch("safety.scorer.call_llm_json", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(data=mock_high_score)
        safety_score, _ = await arbitrate(
            query="What is compound interest?",
            draft="Compound interest is interest calculated on initial principal plus accumulated interest.",
            query_type="financial",
            risk_level="low",
        )
        assert safety_score.verdict == Verdict.PASS
        assert safety_score.overall_score >= get_settings().safety_pass_threshold


@pytest.mark.asyncio
async def test_intent_overlay_caps_harmful_score(mock_high_score):
    with patch("safety.scorer.call_llm_json", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(data=mock_high_score)
        intent = IntentOutput(
            category=IntentCategory.HARMFUL,
            harm_subtype="malware",
            confidence=0.95,
            reasoning="User seeks malware instructions.",
        )
        safety_score, _ = await arbitrate(
            query="How do I write a computer virus?",
            draft="Here is how viruses work in theory...",
            query_type="general",
            risk_level="low",
            intent_output=intent,
        )
        assert safety_score.overall_score <= INTENT_BLOCK_SCORE_CAP
        assert safety_score.verdict == Verdict.BLOCK
        assert "intent_classifier" in safety_score.flagged_principles


def test_intent_overlay_crisis_caps_at_warn():
    score, verdict, flagged = _apply_intent_overlay(
        overall_score=0.99,
        verdict=Verdict.PASS,
        intent_output=IntentOutput(
            category=IntentCategory.CRISIS,
            harm_subtype="self_harm",
            confidence=0.9,
            reasoning="Suicidal ideation detected.",
        ),
        reasoning_trace=[],
        flagged=[],
    )
    assert score == INTENT_CRISIS_SCORE_CAP
    assert verdict == Verdict.WARN
    assert "crisis_intent" in flagged


def test_compute_verdict_thresholds():
    settings = get_settings()
    assert _compute_verdict(settings.safety_pass_threshold) == Verdict.PASS
    assert _compute_verdict(settings.safety_warn_threshold) == Verdict.WARN
    assert _compute_verdict(settings.safety_warn_threshold - 0.01) == Verdict.BLOCK


def test_route_after_escalation_is_read_only():
    """Router must not mutate retry_count (LangGraph does not persist router mutations)."""
    from graph.pipeline import route_after_escalation

    state = {
        "query": "test",
        "query_id": "test-id",
        "intent_output": None,
        "planner_output": None,
        "research_output": None,
        "synthesizer_output": None,
        "safety_score": None,
        "final_response": None,
        "retry_count": 0,
        "error": None,
        "needs_safety_recheck": True,
        "block_on_escalation": False,
        "was_rewritten": True,
        "pipeline_start_ms": 0.0,
    }

    route = route_after_escalation(state)
    assert route == "safety_arbitration"
    assert state["retry_count"] == 0
    assert state["needs_safety_recheck"] is True


@pytest.mark.asyncio
async def test_escalation_retry_cap_blocks_after_max():
    """WARN rewrites must stop after max_safety_retries."""
    from agents.escalation import run_escalation

    settings = get_settings()
    planner = PlannerOutput(
        subtasks=["a", "b"],
        query_type=QueryType.SENSITIVE,
        requires_web_search=False,
        risk_level=RiskLevel.HIGH,
    )
    synthesizer = SynthesizerOutput(draft="draft text", cited_sources=[1], confidence=0.5)
    warn_score = SafetyScore(
        overall_score=0.6,
        principle_scores={"consent_alignment": PrincipleScore(score=0.4, reasoning="misaligned")},
        verdict=Verdict.WARN,
        flagged_principles=["consent_alignment"],
        reasoning_trace=["consent_alignment: misaligned"],
    )

    state = {
        "query": "crisis query",
        "query_id": "retry-test",
        "intent_output": None,
        "planner_output": planner,
        "research_output": None,
        "synthesizer_output": synthesizer,
        "safety_score": warn_score,
        "final_response": None,
        "retry_count": settings.max_safety_retries,
        "error": None,
        "needs_safety_recheck": False,
        "block_on_escalation": False,
        "was_rewritten": True,
        "pipeline_start_ms": 0.0,
    }

    result = await run_escalation(state)
    assert result["final_response"] is not None
    assert result["final_response"].safety_verdict == Verdict.BLOCK.value
    assert result["needs_safety_recheck"] is False


@pytest.mark.asyncio
async def test_escalation_increments_retry_on_rewrite():
    from agents.escalation import run_escalation
    from api.schemas import RewriteLLMResponse

    planner = PlannerOutput(
        subtasks=["a", "b"],
        query_type=QueryType.GENERAL,
        requires_web_search=False,
        risk_level=RiskLevel.MEDIUM,
    )
    synthesizer = SynthesizerOutput(draft="original draft", cited_sources=[], confidence=0.5)
    warn_score = SafetyScore(
        overall_score=0.6,
        principle_scores={"honesty": PrincipleScore(score=0.4, reasoning="issue")},
        verdict=Verdict.WARN,
        flagged_principles=["honesty"],
        reasoning_trace=["honesty: issue"],
    )

    state = {
        "query": "test",
        "query_id": "inc-test",
        "intent_output": None,
        "planner_output": planner,
        "research_output": None,
        "synthesizer_output": synthesizer,
        "safety_score": warn_score,
        "final_response": None,
        "retry_count": 0,
        "error": None,
        "needs_safety_recheck": False,
        "block_on_escalation": False,
        "was_rewritten": False,
        "pipeline_start_ms": 0.0,
    }

    with patch("agents.escalation.call_llm_json", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = LLMResult(
            data=RewriteLLMResponse(rewritten_draft="fixed draft", adjustments="fixed honesty")
        )
        result = await run_escalation(state)

    assert result["retry_count"] == 1
    assert result["needs_safety_recheck"] is True
    assert result["final_response"] is None


def test_all_principles_defined():
    assert len(PRINCIPLES) == 7
    names = {p.name for p in PRINCIPLES}
    assert "non_maleficence" in names
    assert "honesty" in names
    assert HONESTY.weight == 1.5
    assert NON_MALEFICENCE.weight == 1.5
