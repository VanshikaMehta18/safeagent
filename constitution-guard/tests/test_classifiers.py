"""Tests for Constitution Guard classifiers."""

from __future__ import annotations

import re

from constitution_guard.classifiers.crisis import detect_crisis
from constitution_guard.classifiers.pii import PIIClassifier, PII_PATTERNS
from constitution_guard.config import GuardConfig


def test_crisis_detects_self_harm():
    result = detect_crisis("I want to hurt myself")
    assert result.flagged
    assert result.name == "crisis"


def test_crisis_benign_passes():
    result = detect_crisis("What are cold symptoms?")
    assert not result.flagged


def test_pii_detects_email():
    config = GuardConfig(pii_ner=False)
    clf = PIIClassifier(config)
    result = clf.classify("Contact me at user@example.com")
    assert result.flagged
    assert "email" in result.metadata.get("types", [])


def test_pii_benign_passes():
    config = GuardConfig(pii_ner=False)
    clf = PIIClassifier(config)
    result = clf.classify("Hello world")
    assert not result.flagged
