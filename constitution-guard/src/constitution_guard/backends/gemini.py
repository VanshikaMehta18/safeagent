"""Gemini principle critic — reference implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from constitution_guard.backends.critic import PrincipleCritic
from constitution_guard.constitution.principles import ConstitutionalPrinciple
from constitution_guard.models import PrincipleScore, RewriteResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

REWRITE_SYSTEM = (
    "You are a safety editor. Rewrite AI responses to address flagged safety concerns "
    "while preserving helpful, accurate information."
)


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        stripped = "\n".join(lines).strip()
    return stripped


class GeminiCritic:
    """Reference PrincipleCritic implementation using Google Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "Install llm extra: pip install constitution-guard[llm]"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def _call_json(
        self, prompt: str, model: type[T], system: str | None = None
    ) -> T:
        client = self._get_client()
        schema = json.dumps(model.model_json_schema(), indent=2)
        full = f"Respond ONLY with valid JSON matching:\n{schema}\n\n{prompt}"
        from google.genai import types

        response = await client.aio.models.generate_content(
            model=self.model,
            contents=full,
            config=types.GenerateContentConfig(
                system_instruction=system or "Output structured JSON only.",
                max_output_tokens=4096,
                response_mime_type="application/json",
            ),
        )
        text = response.text or "{}"
        return model.model_validate_json(_extract_json(text))

    async def score(
        self,
        principle: ConstitutionalPrinciple,
        query: str,
        draft: str,
        context: dict[str, Any],
    ) -> PrincipleScore:
        prompt = principle.critic_prompt_template.format(query=query, draft=draft)
        return await self._call_json(prompt, PrincipleScore)

    async def rewrite(
        self, query: str, draft: str, flagged: list[str], reasoning_trace: list[str]
    ) -> RewriteResponse:
        prompt = f"""Rewrite this AI draft to address safety concerns: {', '.join(flagged)}

Original query: {query}
Original draft: {draft}

Flagged reasoning:
{chr(10).join(reasoning_trace)}

Return JSON with rewritten_draft and adjustments."""
        return await self._call_json(prompt, RewriteResponse, system=REWRITE_SYSTEM)


# Deprecated alias
class GeminiBackend(GeminiCritic):
    """Deprecated — use GeminiCritic."""

    def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
        super().__init__(api_key=config.gemini_api_key, model=config.gemini_model)


def gemini_critic_from_config(config) -> GeminiCritic:  # type: ignore[no-untyped-def]
    """Build GeminiCritic from GuardConfig fields."""
    return GeminiCritic(api_key=config.gemini_api_key, model=config.gemini_model)
