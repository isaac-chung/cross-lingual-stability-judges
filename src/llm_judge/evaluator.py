"""Core evaluation logic for LLM-as-a-Judge."""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from pydantic import BaseModel, Field

# Handle imports for both module usage and CLI usage
try:
    from ..common.llm_client import create_sync_client
except ImportError:
    # For CLI usage, add parent directory to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.llm_client import create_sync_client
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class ConversationEvaluation(BaseModel):
    """Pydantic model for structured output parsing."""

    G: int = Field(..., ge=0, le=4, description="Grammaticality score (0-4)")
    R: int = Field(..., ge=0, le=4, description="Readability score (0-4)")
    C: int = Field(..., ge=0, le=3, description="Content Coherence score (0-3)")
    F: int = Field(..., ge=0, le=3, description="Fluency score (0-3)")
    explanation: str = Field(..., description="Brief explanation of scores")


class LLMJudge:
    """Evaluator supporting multiple providers (OpenAI, Groq)."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        provider: str = "openai",
        reasoning_effort: str | None = None,
    ):
        """Initialize the judge.

        Args:
            model: Model to use for evaluation.
            provider: API provider ("openai" or "groq").
            reasoning_effort: Reasoning effort for compatible models ("low", "medium", "high").
        """
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.client = self._create_client()

    def _create_client(self):
        """Create unified client based on provider."""
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable required for Groq provider")
            return create_sync_client(
                model=self.model,
                provider="groq",
                api_key=api_key,
            )
        else:
            # OpenAI provider (default)
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://eu.api.openai.com/v1")
            return create_sync_client(
                model=self.model,
                provider="openai",
                api_key=api_key,
                base_url=base_url,
            )

    def format_conversation(self, conv: dict[str, Any]) -> str:
        """Format conversation for prompt (matches original format).

        Args:
            conv: Conversation dict with messages.

        Returns:
            Formatted string for the prompt.
        """
        messages = conv.get("messages", [])
        formatted_lines = []

        for msg in messages:
            from_name = msg.get("from_name", "Unknown")
            from_type = msg.get("from_type", "unknown")
            content = msg.get("message", "")
            formatted_lines.append(f"{from_name} ({from_type}): {content}")

        conversation = "\n".join(formatted_lines)

        return USER_PROMPT_TEMPLATE.format(conversation=conversation)

    def evaluate(self, conv: dict[str, Any]) -> ConversationEvaluation:
        """Evaluate a single conversation.

        Args:
            conv: Conversation dict with subject and messages.

        Returns:
            ConversationEvaluation with scores and explanation.
        """
        user_prompt = self.format_conversation(conv)

        # Build messages list
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text_format": ConversationEvaluation,
        }

        # Add reasoning effort if provided
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        # Retry with exponential backoff
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.responses.parse(**kwargs)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in str(e):
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    raise

        result = response.output_parsed
        if result is None:
            raise ValueError("Failed to parse evaluation response")

        return result

    def evaluate_batch(
        self,
        convs: list[dict[str, Any]],
        workers: int = 5,
        on_complete: Callable[[int, int], None] | None = None,
    ) -> list[ConversationEvaluation]:
        """Concurrent evaluation with ThreadPoolExecutor.

        Args:
            convs: List of conversation dicts.
            workers: Number of concurrent workers.
            on_complete: Optional callback called after each evaluation with (index, total).

        Returns:
            List of ConversationEvaluation results in the same order as input.
        """
        results: list[ConversationEvaluation | None] = [None] * len(convs)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.evaluate, conv): idx
                for idx, conv in enumerate(convs)
            }

            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if on_complete:
                    on_complete(completed, len(convs))

        return results  # type: ignore
