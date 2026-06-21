"""Shared Constitution Guard instance for SafeAgent demo."""

from functools import lru_cache

from constitution_guard import GeminiCritic, Guard, GuardConfig

from config import get_settings


@lru_cache
def get_guard() -> Guard:
    settings = get_settings()
    critic = (
        GeminiCritic(api_key=settings.gemini_api_key, model=settings.gemini_model)
        if settings.gemini_api_key
        else None
    )
    config = GuardConfig(
        jailbreak=True,
        pii=True,
        toxicity=True,
        crisis=True,
        principle_critic=critic,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        pass_threshold=settings.safety_pass_threshold,
        warn_threshold=settings.safety_warn_threshold,
        max_retries=settings.max_safety_retries,
    )
    return Guard(config, critic=critic)
