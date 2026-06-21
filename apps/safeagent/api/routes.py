"""FastAPI route handlers for SafeAgent."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Query

from api.schemas import (
    EvalReport,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SafetyLogEntry,
    SafetyReport,
)
from config import get_settings
from graph.pipeline import run_pipeline
from safety.logger import aggregate_report, read_log_entries

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.app_version)


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    """Run the full LangGraph pipeline for a user query."""
    start = time.perf_counter()
    state = await run_pipeline(body.query)
    latency_ms = (time.perf_counter() - start) * 1000

    final = state.get("final_response")
    if not final:
        from api.schemas import FinalResponse, Verdict

        final = FinalResponse(
            content="Unable to generate a response.",
            safety_verdict=Verdict.BLOCK.value,
            was_rewritten=False,
            referral_suggestion=None,
            safety_note=state.get("error", "Unknown error"),
        )

    return QueryResponse(
        query_id=state["query_id"],
        latency_ms=round(latency_ms, 2),
        response=final,
    )


@router.get("/safety-log", response_model=list[SafetyLogEntry])
async def safety_log(
    verdict: str | None = Query(default=None, description="Filter by verdict"),
) -> list[SafetyLogEntry]:
    """Return the last 50 safety log entries."""
    settings = get_settings()
    entries = read_log_entries(
        limit=settings.safety_log_limit, verdict_filter=verdict
    )
    return [SafetyLogEntry(**e) for e in entries]


@router.get("/safety-report", response_model=SafetyReport)
async def safety_report() -> SafetyReport:
    """Return aggregated safety statistics."""
    report = aggregate_report()
    return SafetyReport(**report)


@router.post("/eval/run", response_model=EvalReport)
async def eval_run() -> EvalReport:
    """Trigger the adversarial eval suite."""
    from evals.eval_runner import run_evals

    report = await run_evals()
    return report
