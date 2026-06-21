"""JSONL safety decision logger."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import get_settings

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()


def hash_query(query: str) -> str:
    """Return SHA-256 hash of a query string."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


async def log_decision(
    *,
    query_id: str,
    query: str,
    query_type: str,
    principle_scores: dict[str, float],
    overall_score: float,
    verdict: str,
    flagged_principles: list[str],
    was_rewritten: bool,
    latency_ms: float,
) -> None:
    """Append a safety arbitration decision to the JSONL log file."""
    settings = get_settings()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_id": query_id,
        "query_hash": hash_query(query),
        "query_type": query_type,
        "principle_scores": principle_scores,
        "overall_score": overall_score,
        "verdict": verdict,
        "flagged_principles": flagged_principles,
        "was_rewritten": was_rewritten,
        "latency_ms": round(latency_ms, 2),
    }

    log_path = Path(settings.safety_log_path)
    async with _lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info("Logged safety decision query_id=%s verdict=%s", query_id, verdict)


def read_log_entries(limit: int = 50, verdict_filter: str | None = None) -> list[dict]:
    """Read the last N entries from the safety log, optionally filtered by verdict."""
    settings = get_settings()
    log_path = Path(settings.safety_log_path)
    if not log_path.exists():
        return []

    entries: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if verdict_filter and entry.get("verdict") != verdict_filter:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed log line")

    return entries[-limit:]


def aggregate_report() -> dict:
    """Compute aggregated safety statistics from the log file."""
    settings = get_settings()
    entries = read_log_entries(limit=10_000)
    if not entries:
        return {
            "total_queries": 0,
            "pass_count": 0,
            "warn_count": 0,
            "block_count": 0,
            "most_flagged_principles": [],
            "average_safety_score": 0.0,
            "average_latency_ms": 0.0,
        }

    pass_count = sum(1 for e in entries if e.get("verdict") == "PASS")
    warn_count = sum(1 for e in entries if e.get("verdict") == "WARN")
    block_count = sum(1 for e in entries if e.get("verdict") == "BLOCK")

    principle_counts: dict[str, int] = {}
    for entry in entries:
        for p in entry.get("flagged_principles", []):
            principle_counts[p] = principle_counts.get(p, 0) + 1

    top_principles = sorted(principle_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    most_flagged = [{"principle": p, "count": c} for p, c in top_principles]

    scores = [e.get("overall_score", 0.0) for e in entries]
    latencies = [e.get("latency_ms", 0.0) for e in entries]

    return {
        "total_queries": len(entries),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "block_count": block_count,
        "most_flagged_principles": most_flagged,
        "average_safety_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
    }
