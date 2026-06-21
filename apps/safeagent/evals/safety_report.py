"""Generate markdown safety eval reports."""

from __future__ import annotations

import statistics
from pathlib import Path

from api.schemas import EvalReport

REPORT_PATH = Path("eval_report.md")


def generate_report(report: EvalReport) -> str:
    """Generate eval_report.md from eval results."""
    false_negatives = [
        r for r in report.results
        if r.category in ("harmful_requests", "jailbreak_attempts", "manipulation_attempts")
        and r.actual_verdict == "PASS"
    ]

    latencies = [r.latency_ms for r in report.results]
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (
        sorted(latencies)[int(len(latencies) * 0.95)]
        if latencies
        else 0.0
    )

    principle_flags: dict[str, int] = {}
    for r in report.results:
        if r.actual_verdict in ("WARN", "BLOCK"):
            principle_flags[r.category] = principle_flags.get(r.category, 0) + 1

    top_flagged = sorted(principle_flags.items(), key=lambda x: x[1], reverse=True)[:3]

    lines = [
        "# SafeAgent Eval Report",
        "",
        "## Overall Accuracy",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total prompts | {report.total} |",
        f"| Accuracy | {report.accuracy * 100:.1f}% |",
        f"| False positive rate | {report.false_positive_rate * 100:.1f}% |",
        f"| False negative rate | {report.false_negative_rate * 100:.1f}% |",
        f"| Mean latency | {report.mean_latency_ms:.0f}ms |",
        "",
        "## Per-Category Breakdown",
        "",
        "| Category | Accuracy | Mean Latency | Total |",
        "|----------|----------|--------------|-------|",
    ]

    for cat, stats in report.per_category.items():
        lines.append(
            f"| {cat} | {stats['accuracy'] * 100:.1f}% | "
            f"{stats['mean_latency_ms']:.0f}ms | {int(stats['total'])} |"
        )

    lines.extend([
        "",
        "## Top Flagged Categories",
        "",
    ])
    for cat, count in top_flagged:
        lines.append(f"- **{cat}**: {count} flagged responses")

    lines.extend([
        "",
        "## Worst False Negatives (Critical)",
        "",
    ])
    if false_negatives:
        for fn in false_negatives[:5]:
            lines.append(f"- `{fn.prompt[:80]}...` (expected {fn.expected_verdict}, got {fn.actual_verdict})")
    else:
        lines.append("_No false negatives detected._")

    lines.extend([
        "",
        "## Latency Analysis",
        "",
        f"- Safety layer p50: {p50:.0f}ms",
        f"- Safety layer p95: {p95:.0f}ms",
        "",
    ])

    content = "\n".join(lines)
    REPORT_PATH.write_text(content, encoding="utf-8")
    return content
