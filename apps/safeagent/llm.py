"""Async Gemini client wrapper with structured JSON output."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMUsage:
    """Token usage metadata from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class LLMResult:
    """Structured result from an LLM JSON call."""

    data: BaseModel
    usage: LLMUsage = field(default_factory=LLMUsage)


_client: genai.Client | None = None
_last_usage: LLMUsage = LLMUsage()


def get_client() -> genai.Client:
    """Return a shared Gemini client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def get_last_usage() -> LLMUsage:
    """Return token usage from the most recent LLM call."""
    return _last_usage


def _build_json_prompt(prompt: str, response_model: type[BaseModel], strict: bool = False) -> str:
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    prefix = (
        "You must respond with valid JSON only. No markdown, no explanation outside JSON.\n"
        if strict
        else "Respond ONLY with a valid JSON object matching this schema:\n"
    )
    return f"{prefix}{schema}\n\n{prompt}"


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _usage_from_response(response: types.GenerateContentResponse, latency_ms: float) -> LLMUsage:
    meta = response.usage_metadata
    if meta is None:
        return LLMUsage(latency_ms=latency_ms)
    return LLMUsage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        latency_ms=latency_ms,
    )


async def call_llm_json(
    prompt: str,
    response_model: type[T],
    *,
    system: str | None = None,
    max_tokens: int = 4096,
) -> LLMResult:
    """
    Call Gemini and parse the response into a Pydantic model.

    Retries once with a stricter JSON prompt on parse failure.
    """
    global _last_usage
    settings = get_settings()
    client = get_client()

    for attempt in range(2):
        full_prompt = _build_json_prompt(prompt, response_model, strict=attempt > 0)
        start = time.perf_counter()
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system
                    or "You are a helpful assistant that outputs structured JSON.",
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.text or ""
            parsed = response_model.model_validate_json(_extract_json(text))
            usage = _usage_from_response(response, latency_ms)
            _last_usage = usage
            return LLMResult(data=parsed, usage=usage)
        except Exception as exc:
            logger.warning("Gemini JSON parse attempt %d failed: %s", attempt + 1, exc)
            if attempt == 1:
                raise

    raise RuntimeError("Unreachable")


async def call_llm_text(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 4096,
) -> tuple[str, LLMUsage]:
    """Call Gemini and return raw text with usage metadata."""
    global _last_usage
    settings = get_settings()
    client = get_client()

    start = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system or "You are a helpful assistant.",
            max_output_tokens=max_tokens,
        ),
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.text or ""
    usage = _usage_from_response(response, latency_ms)
    _last_usage = usage
    return text, usage


# Backward-compatible aliases (deprecated)
call_claude_json = call_llm_json
call_claude_text = call_llm_text
