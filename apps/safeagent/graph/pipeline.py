"""LangGraph pipeline orchestrating the SafeAgent safety flow."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Literal

from langgraph.graph import END, StateGraph

from agents.escalation import run_escalation
from agents.intent_classifier import run_intent_classifier
from agents.intent_handlers import run_crisis_handler, run_harmful_block
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.safety_arbitration import run_safety_arbitration
from agents.synthesizer import run_synthesizer
from api.schemas import FinalResponse, IntentCategory, Verdict
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _initial_state(query: str, query_id: str | None = None) -> AgentState:
    return AgentState(
        query=query,
        query_id=query_id or str(uuid.uuid4()),
        intent_output=None,
        planner_output=None,
        research_output=None,
        synthesizer_output=None,
        safety_score=None,
        final_response=None,
        retry_count=0,
        error=None,
        needs_safety_recheck=False,
        block_on_escalation=False,
        was_rewritten=False,
        pipeline_start_ms=time.perf_counter() * 1000,
    )


async def safe_fallback_node(state: AgentState) -> AgentState:
    """Return a safe fallback response when any agent fails."""
    error_msg = state.get("error", "An unexpected error occurred")
    logger.error("Pipeline error, returning safe fallback: %s", error_msg)
    state["final_response"] = FinalResponse(
        content=(
            "I'm sorry, I encountered an issue processing your request. "
            "Please try rephrasing your question or consult a qualified professional "
            "for time-sensitive matters."
        ),
        safety_verdict=Verdict.BLOCK.value,
        was_rewritten=False,
        referral_suggestion="a qualified professional in the relevant field",
        safety_note=f"Pipeline error: {error_msg}",
    )
    return state


def route_after_intent(
    state: AgentState,
) -> Literal["harmful_block", "crisis_handler", "planner", "safe_fallback"]:
    """Route based on upstream intent classification (read-only)."""
    if state.get("error") and not state.get("intent_output"):
        return "safe_fallback"

    intent = state.get("intent_output")
    if not intent:
        return "planner"

    if intent.category == IntentCategory.CRISIS:
        return "crisis_handler"

    if intent.category in (IntentCategory.HARMFUL, IntentCategory.JAILBREAK):
        return "harmful_block"

    return "planner"


def route_after_escalation(
    state: AgentState,
) -> Literal["safety_arbitration", "escalation", "safe_fallback", "__end__"]:
    """Route after escalation: re-check rewritten drafts or end (read-only)."""
    if state.get("error") and not state.get("final_response"):
        return "safe_fallback"

    if state.get("needs_safety_recheck"):
        if state.get("block_on_escalation"):
            return "escalation"
        return "safety_arbitration"

    return "__end__"


def route_after_safety(
    state: AgentState,
) -> Literal["escalation", "safe_fallback"]:
    """Route to escalation or fallback after safety arbitration."""
    if state.get("error") and not state.get("safety_score"):
        return "safe_fallback"
    return "escalation"


def build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the SafeAgent pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", run_intent_classifier)
    graph.add_node("harmful_block", run_harmful_block)
    graph.add_node("crisis_handler", run_crisis_handler)
    graph.add_node("planner", run_planner)
    graph.add_node("researcher", run_researcher)
    graph.add_node("synthesizer", run_synthesizer)
    graph.add_node("safety_arbitration", run_safety_arbitration)
    graph.add_node("escalation", run_escalation)
    graph.add_node("safe_fallback", safe_fallback_node)

    graph.set_entry_point("intent_classifier")
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "harmful_block": "harmful_block",
            "crisis_handler": "crisis_handler",
            "planner": "planner",
            "safe_fallback": "safe_fallback",
        },
    )
    graph.add_edge("harmful_block", END)
    graph.add_edge("crisis_handler", END)
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "safety_arbitration")

    graph.add_conditional_edges(
        "safety_arbitration",
        route_after_safety,
        {"escalation": "escalation", "safe_fallback": "safe_fallback"},
    )

    graph.add_conditional_edges(
        "escalation",
        route_after_escalation,
        {
            "safety_arbitration": "safety_arbitration",
            "escalation": "escalation",
            "safe_fallback": "safe_fallback",
            "__end__": END,
        },
    )

    graph.add_edge("safe_fallback", END)
    return graph


_compiled_graph = None


def get_compiled_graph():
    """Return compiled graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def reset_compiled_graph() -> None:
    """Clear cached graph (for tests)."""
    global _compiled_graph
    _compiled_graph = None


async def run_pipeline(query: str, query_id: str | None = None) -> AgentState:
    """Run the full SafeAgent pipeline for a user query."""
    graph = get_compiled_graph()
    state = _initial_state(query, query_id)
    result = await graph.ainvoke(state)

    if not result.get("final_response") and result.get("error"):
        result = await safe_fallback_node(result)

    return result
