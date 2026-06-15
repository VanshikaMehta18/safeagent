"""LangSmith tracing configuration and helpers."""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from config import get_settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_traceable_available = False

try:
    from langsmith import traceable as _ls_traceable

    _traceable_available = True
except ImportError:
    _ls_traceable = None


def configure_langsmith() -> None:
    """Set LangSmith environment variables from settings."""
    settings = get_settings()
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project %s", settings.langsmith_project)


def traceable(name: str) -> Callable[[F], F]:
    """Decorator that wraps LangSmith traceable with a consistent agent name."""

    def decorator(fn: F) -> F:
        if _traceable_available and _ls_traceable is not None:
            traced = _ls_traceable(name=name, run_type="chain")(fn)
            return traced  # type: ignore[return-value]

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("Agent %s completed in %.1fms", name, elapsed)
                return result
            except Exception:
                logger.exception("Agent %s failed", name)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
