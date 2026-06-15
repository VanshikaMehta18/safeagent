"""Synthesizer agent — drafts a cited response from research."""

from __future__ import annotations

import logging

from api.schemas import SynthesizerLLMResponse, SynthesizerOutput
from graph.state import AgentState
from llm import call_llm_json
from observability.langsmith_tracer import traceable

logger = logging.getLogger(__name__)

SYNTHESIZER_SYSTEM = (
    "You are a research synthesizer. Write clear, well-cited responses "
    "based on provided sources. Always cite sources inline using [1], [2], etc."
)


@traceable("safeagent.synthesizer")
async def run_synthesizer(state: AgentState) -> AgentState:
    """
    Synthesize a draft response with inline citations from research sources.

    Output goes to Safety Arbitration — never directly to the user.
    """
    if state.get("error") or not state.get("research_output"):
        return state

    query = state["query"]
    sources = state["research_output"].sources

    sources_text = "\n\n".join(
        f"[{i + 1}] {s.content}\nURL: {s.url} (relevance: {s.relevance_score:.2f})"
        for i, s in enumerate(sources)
    )

    prompt = f"""Write a comprehensive draft response to the user's query using the research sources.

User query: {query}

Research sources:
{sources_text if sources_text else "No sources available — provide a cautious general response."}

Requirements:
- Cite sources inline using [1], [2], etc.
- Be accurate and acknowledge uncertainty where sources are limited
- Return JSON with: draft (string), cited_sources (list of int indices), confidence (0.0-1.0)
"""

    try:
        result = await call_llm_json(
            prompt, SynthesizerLLMResponse, system=SYNTHESIZER_SYSTEM
        )
        raw = result.data
        state["synthesizer_output"] = SynthesizerOutput(
            draft=raw.draft,
            cited_sources=raw.cited_sources,
            confidence=max(0.0, min(1.0, raw.confidence)),
        )
        logger.info(
            "Synthesizer: draft length=%d citations=%s",
            len(raw.draft),
            raw.cited_sources,
        )
    except Exception as exc:
        logger.exception("Synthesizer agent failed")
        state["error"] = f"Synthesizer failed: {exc}"

    return state
