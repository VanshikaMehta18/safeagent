"""Constitution Guard — pip-installable Constitutional AI middleware."""

from constitution_guard.backends.critic import PrincipleCritic
from constitution_guard.backends.gemini import GeminiCritic
from constitution_guard.config import GuardConfig
from constitution_guard.guard import Guard
from constitution_guard.models import CheckResult, GuardResult, SafetyScore, Verdict

__version__ = "0.1.0"
__all__ = [
    "Guard",
    "GuardConfig",
    "GuardResult",
    "CheckResult",
    "SafetyScore",
    "Verdict",
    "PrincipleCritic",
    "GeminiCritic",
]
