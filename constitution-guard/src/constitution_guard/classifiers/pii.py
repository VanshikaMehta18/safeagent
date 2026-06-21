"""PII detection via regex (optional NER)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from constitution_guard.config import GuardConfig
from constitution_guard.models import CheckResult

logger = logging.getLogger(__name__)

_ner_pipeline: Any = None

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
    ("api_key", re.compile(r"\b(?:sk-|AKIA|AIza)[A-Za-z0-9_-]{10,}\b")),
]


def _get_ner(config: GuardConfig):
    global _ner_pipeline
    if _ner_pipeline is not None:
        return _ner_pipeline
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Install classifiers extra: pip install constitution-guard[classifiers]"
        ) from exc
    _ner_pipeline = pipeline(
        "ner",
        model=config.ner_model,
        aggregation_strategy="simple",
        device=-1 if config.device == "cpu" else 0,
    )
    return _ner_pipeline


class PIIClassifier:
    name = "pii"

    def __init__(self, config: GuardConfig) -> None:
        self.config = config

    def classify(self, text: str) -> CheckResult:
        start = time.perf_counter()
        found: list[str] = []

        for label, pattern in PII_PATTERNS:
            if pattern.search(text):
                found.append(label)

        if self.config.pii_ner and len(text) < 2000:
            try:
                ner = _get_ner(self.config)
                entities = ner(text)
                for ent in entities:
                    if ent.get("entity_group") in ("PER", "LOC"):
                        found.append(ent["entity_group"].lower())
            except Exception as exc:
                logger.debug("NER skipped: %s", exc)

        flagged = len(found) > 0
        safe_score = 0.3 if flagged else 1.0
        reasoning = (
            f"PII detected: {', '.join(set(found))}" if flagged else "No PII detected"
        )
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name=self.name,
            score=safe_score,
            flagged=flagged,
            reasoning=reasoning,
            latency_ms=latency,
            metadata={"types": list(set(found))},
        )
