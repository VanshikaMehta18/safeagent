"""Tests for constitutional safety scoring (via constitution-guard)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from api.schemas import PlannerOutput, QueryType, RiskLevel, Verdict
from config import get_settings
from constitution_guard.config import GuardConfig
from constitution_guard.constitution.principles import HONESTY, NON_MALEFICENCE, PRINCIPLES
from constitution_guard.constitution.scorer import compute_verdict
from constitution_guard.models import PrincipleScore, SafetyScore
from graph.pipeline import route_after_escalation


@pytest.fixture
def mock_high_score():
    return PrincipleScore(score=0.95, reasoning="Response is safe.")


@pytest.fixture
def mock_low_score():
    return PrincipleScore(score=0.2, reasoning="Response could cause harm.")


def test_compute_verdict_thresholds():
    config = GuardConfig(
        pass_threshold=get_settings().safety_pass_threshold,
        warn_threshold=get_settings().safety_warn_threshold,
    )
    assert compute_verdict(config.pass_threshold, config) == Verdict.PASS
    assert compute_verdict(config.warn_threshold, config) == Verdict.WARN
    assert compute_verdict(config.warn_threshold - 0.01, config) == Verdict.BLOCK


def test_route_after_escalation_is_read_only():
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


@pytest.mark.asyncio
async def test_escalation_retry_cap_blocks_after_max():
    from agents.escalation import run_escalation
    from api.schemas import SynthesizerOutput

    settings = get_settings()
    planner = PlannerOutput(
        subtasks=["a", "b"],
        query_type=QueryType.SENSITIVE,
        requires_web_search=False,
        risk_level=RiskLevel.HIGH,
    )
    synthesizer = SynthesizerOutput(draft="draft", cited_sources=[1], confidence=0.5)
    warn_score = SafetyScore(
        overall_score=0.6,
        principle_scores={"fairness": PrincipleScore(score=0.4, reasoning="x")},
        verdict=Verdict.WARN,
        flagged_principles=["fairness"],
        reasoning_trace=["fairness: x"],
    )
    state = {
        "query": "test",
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
    assert result["final_response"].safety_verdict == Verdict.BLOCK.value


def test_all_principles_defined():
    assert len(PRINCIPLES) == 7
    assert HONESTY.weight == 1.5
    assert NON_MALEFICENCE.weight == 1.5
