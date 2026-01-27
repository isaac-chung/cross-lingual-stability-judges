"""Minimal conversation generator for synthetic customer support conversations."""

from .config import (
    INDUSTRIES,
    LANGUAGES,
    PROBLEMS,
    ConversationConfig,
    random_config,
)
from .generator import ConversationGenerator, PromptPart

__all__ = [
    "ConversationGenerator",
    "PromptPart",
    "random_config",
    "ConversationConfig",
    "LANGUAGES",
    "INDUSTRIES",
    "PROBLEMS",
]
