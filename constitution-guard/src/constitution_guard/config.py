"""Guard configuration."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GuardConfig(BaseSettings):
    """Configuration for Constitution Guard middleware."""

    model_config = SettingsConfigDict(extra="ignore", arbitrary_types_allowed=True)

    jailbreak: bool = True
    pii: bool = True
    toxicity: bool = True
    crisis: bool = True
    pii_ner: bool = False

    jailbreak_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    toxicity_model: str = "unitary/toxic-bert"
    ner_model: str = "Jean-Baptiste/roberta-base-ner-english"

    jailbreak_block_threshold: float = 0.5
    toxicity_warn_threshold: float = 0.7
    toxicity_block_threshold: float = 0.9

    pass_threshold: float = 0.8
    warn_threshold: float = 0.5
    max_retries: int = 2

    principle_critic: Any | None = Field(default=None, exclude=True)
    constitutional_backend: str | None = None  # deprecated: use principle_critic
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")

    device: str = "cpu"
    return_content: bool = False

    query_type: str = "general"
    risk_level: str = "medium"
