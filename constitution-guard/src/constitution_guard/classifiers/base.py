"""Classifier protocol."""

from __future__ import annotations

from typing import Protocol

from constitution_guard.models import CheckResult


class ClassifierProtocol(Protocol):
    name: str

    def classify(self, text: str) -> CheckResult: ...
