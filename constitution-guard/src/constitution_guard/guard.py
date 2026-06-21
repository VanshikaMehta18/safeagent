"""Guard — main middleware entry point."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from constitution_guard.backends.critic import PrincipleCritic
from constitution_guard.config import GuardConfig
from constitution_guard.models import GuardChecks, GuardResult
from constitution_guard.pipeline import (
    has_llm_critic,
    resolve_critic,
    run_checks_async,
    run_guarded_pipeline,
    _build_classifiers,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class Guard:
    """Constitutional AI middleware — wrap any agent with safety checks."""

    def __init__(
        self,
        config: GuardConfig | None = None,
        critic: PrincipleCritic | None = None,
    ) -> None:
        self.config = config or GuardConfig()
        self._critic = resolve_critic(self.config, critic)
        self._classifiers = _build_classifiers(self.config)

    @property
    def critic(self) -> PrincipleCritic | None:
        return self._critic

    async def check_input(self, text: str) -> GuardChecks:
        return await run_checks_async(text, self._classifiers, self.config)

    async def check_output(self, query: str, draft: str) -> GuardChecks:
        return await run_checks_async(draft, self._classifiers, self.config)

    async def arbitrate(self, query: str, draft: str) -> GuardResult:
        from constitution_guard.constitution.scorer import arbitrate as _arb
        from constitution_guard.constitution.escalation import apply_escalation

        output_checks = await self.check_output(query, draft)
        score, _ = await _arb(query, draft, output_checks, self._critic, self.config)

        if not has_llm_critic(self.config, self._critic):
            return GuardResult(
                content=draft,
                verdict=score.verdict,
                input_checks=GuardChecks(),
                output_checks=output_checks,
                constitutional_score=score,
                safety_note="Local guardrails; LLM principles unscored.",
            )

        content, verdict, referral, note, _ = apply_escalation(
            query, draft, score, self.config
        )
        return GuardResult(
            content=content,
            verdict=verdict,
            input_checks=GuardChecks(),
            output_checks=output_checks,
            constitutional_score=score,
            referral=referral,
            safety_note=note,
        )

    async def run(self, query: str, agent_fn: Callable[..., str | Awaitable[str]]) -> GuardResult:
        return await run_guarded_pipeline(query, agent_fn, self.config, self._critic)

    def wrap(self, fn: F) -> F:
        """Decorator: wrap any sync or async (query) -> str agent."""

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> GuardResult | str:
            query = args[0] if args else kwargs.get("query", "")
            async def agent(q: str) -> str:
                import inspect
                if inspect.iscoroutinefunction(fn):
                    return str(await fn(q, **{k: v for k, v in kwargs.items() if k != "query"}))
                return str(fn(q, **{k: v for k, v in kwargs.items() if k != "query"}))

            result = await run_guarded_pipeline(str(query), agent, self.config, self._critic)
            if self.config.return_content:
                return result.content
            return result

        return async_wrapper  # type: ignore[return-value]

    def wrap_node(self, fn: F) -> F:
        """Wrap a LangGraph state node — expects state dict with 'query' key."""
        from constitution_guard.integrations.langgraph import wrap_node

        return wrap_node(fn, self)  # type: ignore[return-value]
