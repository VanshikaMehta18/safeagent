"""Local jailbreak / prompt-injection classifier."""

from __future__ import annotations

import logging
import time
from typing import Any

from constitution_guard.config import GuardConfig
from constitution_guard.models import CheckResult

logger = logging.getLogger(__name__)

_pipeline: Any = None
_model_id: str | None = None


def _get_pipeline(config: GuardConfig):
    global _pipeline, _model_id
    if _pipeline is not None and _model_id == config.jailbreak_model:
        return _pipeline
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Install classifiers extra: pip install constitution-guard[classifiers]"
        ) from exc
    logger.info("Loading jailbreak model %s", config.jailbreak_model)
    _pipeline = pipeline(
        "text-classification",
        model=config.jailbreak_model,
        device=-1 if config.device == "cpu" else 0,
        truncation=True,
        max_length=512,
    )
    _model_id = config.jailbreak_model
    return _pipeline


class JailbreakClassifier:
    name = "jailbreak"

    def __init__(self, config: GuardConfig) -> None:
        self.config = config

    def classify(self, text: str) -> CheckResult:
        start = time.perf_counter()
        try:
            pipe = _get_pipeline(self.config)
            result = pipe(text[:2000])[0]
            label = result.get("label", "").upper()
            raw_score = float(result.get("score", 0.0))
            is_injection = "INJECTION" in label or label == "LABEL_1"
            risk = raw_score if is_injection else 1.0 - raw_score
            safe_score = 1.0 - risk
            flagged = risk >= self.config.jailbreak_block_threshold
            reasoning = (
                f"Prompt injection detected (risk={risk:.2f})"
                if flagged
                else f"No jailbreak detected (risk={risk:.2f})"
            )
        except Exception as exc:
            logger.warning("Jailbreak classifier failed: %s", exc)
            safe_score = 0.5
            flagged = False
            reasoning = f"Classifier unavailable: {exc}"

        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name=self.name,
            score=safe_score,
            flagged=flagged,
            reasoning=reasoning,
            latency_ms=latency,
        )
