"""Pluggable principle critic protocol."""

from __future__ import annotations

from typing import Any, Protocol

from constitution_guard.constitution.principles import ConstitutionalPrinciple
from constitution_guard.models import PrincipleScore, RewriteResponse


class PrincipleCritic(Protocol):
    """Provider-agnostic interface for LLM-backed constitutional critics."""

    async def score(
        self,
        principle: ConstitutionalPrinciple,
        query: str,
        draft: str,
        context: dict[str, Any],
    ) -> PrincipleScore: ...

    async def rewrite(
        self, query: str, draft: str, flagged: list[str], reasoning_trace: list[str]
    ) -> RewriteResponse: ...
