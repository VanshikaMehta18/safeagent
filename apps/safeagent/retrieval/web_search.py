"""Async web search via DuckDuckGo."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _sync_search(query: str, max_results: int) -> list[dict[str, Any]]:
    from ddgs import DDGS

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for i, item in enumerate(ddgs.text(query, max_results=max_results)):
            results.append(
                {
                    "content": item.get("body", item.get("snippet", "")),
                    "url": item.get("href", item.get("link", "")),
                    "relevance_score": max(0.1, 1.0 - (i * 0.15)),
                }
            )
    return results


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Run an async DuckDuckGo web search.

    Returns a list of dicts with content, url, and relevance_score.
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _sync_search, query, max_results
        )
    except Exception as exc:
        logger.warning("Web search failed for query '%s': %s", query[:50], exc)
        return []
