"""Constitutional backend protocol (deprecated — use PrincipleCritic)."""

from __future__ import annotations

from constitution_guard.backends.critic import PrincipleCritic

# Backward-compatible alias
ConstitutionalBackend = PrincipleCritic

__all__ = ["ConstitutionalBackend", "PrincipleCritic"]
