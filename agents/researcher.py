"""Researcher agent — retrieves sources via FAISS and web search."""

from __future__ import annotations

import hashlib
import logging

from api.schemas import ResearchOutput, Source
from config import get_settings
from graph.state import AgentState
from observability.langsmith_tracer import traceable
from retrieval.vector_store import VectorStore
from retrieval.web_search import search_web

logger = logging.getLogger(__name__)

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return shared vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _dedupe_key(source: Source) -> str:
    content_hash = hashlib.md5(source.content[:200].encode()).hexdigest()
    return f"{source.url}:{content_hash}"


def _deduplicate_sources(sources: list[Source]) -> list[Source]:
    seen: set[str] = set()
    unique: list[Source] = []
    for src in sources:
        key = _dedupe_key(src)
        if key not in seen:
            seen.add(key)
            unique.append(src)
    return unique


@traceable("safeagent.researcher")
async def run_researcher(state: AgentState) -> AgentState:
    """
    Gather sources for each planner subtask via FAISS retrieval and optional web search.

    Deduplicates sources before returning to the synthesizer.
    """
    if state.get("error") or not state.get("planner_output"):
        return state

    settings = get_settings()
    planner = state["planner_output"]
    store = get_vector_store()
    all_sources: list[Source] = []

    try:
        for subtask in planner.subtasks:
            faiss_results = store.search(subtask, k=settings.faiss_top_k)
            for r in faiss_results:
                all_sources.append(
                    Source(
                        content=r.content,
                        url=r.metadata.get("url", "internal://faiss"),
                        relevance_score=min(1.0, max(0.0, r.score)),
                    )
                )

            needs_web = planner.requires_web_search or len(faiss_results) < settings.min_faiss_hits
            if needs_web:
                web_results = await search_web(
                    subtask, max_results=settings.web_search_max_results
                )
                for wr in web_results:
                    all_sources.append(
                        Source(
                            content=wr["content"],
                            url=wr["url"],
                            relevance_score=wr["relevance_score"],
                        )
                    )

        deduped = _deduplicate_sources(all_sources)
        state["research_output"] = ResearchOutput(sources=deduped)
        logger.info("Researcher: gathered %d unique sources", len(deduped))
    except Exception as exc:
        logger.exception("Researcher agent failed")
        state["error"] = f"Researcher failed: {exc}"

    return state
