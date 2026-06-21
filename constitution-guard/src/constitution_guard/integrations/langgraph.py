"""LangGraph integration."""

from __future__ import annotations

import functools
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from constitution_guard.guard import Guard


def wrap_node(fn, guard: "Guard"):
    """Wrap a LangGraph node function that receives/returns state dict."""

    @functools.wraps(fn)
    async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")

        input_checks = await guard.check_input(query)
        state["guard_input_checks"] = input_checks.model_dump()

        if input_checks.verdict.value == "BLOCK":
            from constitution_guard.models import Verdict
            from constitution_guard.classifiers.crisis import CRISIS_MESSAGE

            if any(r.name == "crisis" and r.flagged for r in input_checks.results):
                state["final_response"] = {
                    "content": CRISIS_MESSAGE,
                    "safety_verdict": Verdict.WARN.value,
                    "was_rewritten": False,
                    "referral_suggestion": "988 Suicide & Crisis Lifeline",
                    "safety_note": "Crisis response.",
                }
            else:
                state["final_response"] = {
                    "content": "Request blocked by Constitution Guard input gate.",
                    "safety_verdict": Verdict.BLOCK.value,
                    "was_rewritten": False,
                    "referral_suggestion": None,
                    "safety_note": "Input blocked.",
                }
            return state

        state = await fn(state)
        return state

    return wrapper
