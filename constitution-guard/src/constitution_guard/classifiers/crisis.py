"""Crisis / self-harm keyword heuristics (no LLM)."""

from __future__ import annotations

import re
import time

from constitution_guard.models import CheckResult, Verdict

CRISIS_MESSAGE = (
    "I'm concerned about what you're going through, and you don't have to face this alone. "
    "Please reach out for help — call or text 988 (Suicide & Crisis Lifeline), available 24/7. "
    "If you are in immediate danger, call 911 or go to your nearest emergency room. "
    "Talking to a counselor, trusted friend, or family member can make a real difference."
)

CRISIS_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(kill|hurt|harm)\s+(my)?self\b",
        r"\bsuicid(e|al)\b",
        r"\b(end|take)\s+my\s+life\b",
        r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
        r"\bself[\-\s]?harm\b",
        r"\bwant\s+to\s+die\b",
    ]
]


def detect_crisis(text: str) -> CheckResult:
    start = time.perf_counter()
    matched = [p.pattern for p in CRISIS_PATTERNS if p.search(text)]
    flagged = len(matched) > 0
    latency = (time.perf_counter() - start) * 1000
    return CheckResult(
        name="crisis",
        score=0.5 if flagged else 1.0,
        flagged=flagged,
        reasoning="Crisis/self-harm language detected" if flagged else "No crisis indicators",
        latency_ms=latency,
        metadata={"patterns": matched},
    )


def crisis_verdict() -> Verdict:
    return Verdict.WARN
