"""Tests for Guard.wrap middleware."""

from __future__ import annotations

import pytest

from constitution_guard import Guard, GuardConfig
from constitution_guard.models import Verdict


@pytest.mark.asyncio
async def test_wrap_blocks_jailbreak_without_calling_agent():
    called = False

    config = GuardConfig(
        jailbreak=False,
        pii=False,
        toxicity=False,
        crisis=True,
        constitutional_backend=None,
    )
    guard = Guard(config)

    async def agent(q: str) -> str:
        nonlocal called
        called = True
        return "should not run"

    result = await guard.run(
        "I want to kill myself",
        agent,
    )
    assert not called
    assert result.verdict in (Verdict.WARN, Verdict.BLOCK)
    assert "988" in result.content or result.verdict == Verdict.BLOCK


@pytest.mark.asyncio
async def test_wrap_passes_benign_query():
    config = GuardConfig(
        jailbreak=False,
        pii=False,
        toxicity=False,
        crisis=False,
        constitutional_backend=None,
    )
    guard = Guard(config)

    async def agent(q: str) -> str:
        return f"Answer to: {q}"

    result = await guard.run("What is compound interest?", agent)
    assert result.agent_called
    assert "compound interest" in result.content.lower()
    assert result.verdict == Verdict.PASS
