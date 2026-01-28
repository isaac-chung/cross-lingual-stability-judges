"""
Unified LLM client abstraction using liteLLM.

This module provides a drop-in replacement for OpenAI clients while using
liteLLM as the backend. It preserves the exact API patterns used in the
existing codebase while enabling multi-provider support.
"""

import asyncio
import json
import random
import time
from typing import Any, Dict, List, Optional, Type, Union

import instructor
import litellm
from litellm import completion, acompletion
from pydantic import BaseModel


class MockResponse:
    """
    Maintains backward compatibility with OpenAI's response.output_parsed pattern.

    This allows existing code using `response.output_parsed` to continue working
    without modification.
    """

    def __init__(self, output_parsed: Any = None):
        self.output_parsed = output_parsed


class MockChatCompletion:
    """
    Mock object to maintain compatibility with OpenAI's ChatCompletion structure.
    """

    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice object for ChatCompletion compatibility."""

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message object for ChatCompletion compatibility."""

    def __init__(self, content: str):
        self.content = content


class UnifiedLLMClient:
    """
    Unified client that wraps liteLLM and provides OpenAI-compatible interface.

    This client provides drop-in replacements for:
    1. OpenAI's responses.parse() API using instructor
    2. OpenAI's chat.completions.create() API using liteLLM
    3. AsyncOpenAI's async methods
    4. Multi-provider support (OpenAI, Groq, etc.)
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the unified LLM client.

        Args:
            model: The model name (e.g., "gpt-4o-mini", "llama-3.1-70b-versatile")
            provider: Provider name ("openai", "groq", "anthropic", etc.)
            api_key: API key for the provider
            base_url: Base URL for the provider API
            **kwargs: Additional configuration options
        """
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self._configure_litellm(**kwargs)

        # Create instructor client for structured output
        self.instructor_client = instructor.from_litellm(completion)

    def _configure_litellm(self, **kwargs):
        """Configure liteLLM with provider-specific settings."""
        # Set API keys based on provider
        if self.api_key:
            if self.provider == "openai":
                litellm.openai_key = self.api_key
            elif self.provider == "groq":
                litellm.groq_key = self.api_key
            elif self.provider == "anthropic":
                litellm.anthropic_key = self.api_key

        # Set base URL if provided
        if self.base_url and self.provider == "openai":
            litellm.api_base = self.base_url

        # Configure provider-specific settings
        litellm.drop_params = True  # Drop unsupported parameters automatically
        litellm.set_verbose = False  # Reduce noise in logs

    def _format_model_name(self, **kwargs):
        """
        Format model name for liteLLM based on provider.

        Args:
            **kwargs: Keyword arguments including model name

        Returns:
            Updated kwargs with properly formatted model name
        """
        kwargs = kwargs.copy()  # Don't modify original
        model = kwargs.get("model", self.model)

        if self.provider == "groq":
            # For Groq, prefix with provider if not already prefixed
            if not model.startswith("groq/"):
                kwargs["model"] = f"groq/{model}"
        elif self.provider == "anthropic":
            # For Anthropic, prefix with provider if not already prefixed
            if not model.startswith("anthropic/"):
                kwargs["model"] = f"anthropic/{model}"
        # For OpenAI, use model name as-is

        return kwargs

    def responses_parse(
        self,
        model: str,
        input: List[Dict[str, str]],
        text_format: Type[BaseModel],
        reasoning_effort: Optional[str] = None,
        **kwargs
    ) -> MockResponse:
        """
        Drop-in replacement for OpenAI's responses.parse() using instructor.

        Args:
            model: Model name
            input: List of message dictionaries
            text_format: Pydantic model class for structured output
            reasoning_effort: Reasoning effort level (for compatible models)
            **kwargs: Additional parameters

        Returns:
            MockResponse with output_parsed attribute containing validated Pydantic object
        """
        # Build messages for liteLLM
        messages = input.copy()

        # Handle reasoning effort parameter
        extra_kwargs = {}
        if reasoning_effort:
            extra_kwargs["reasoning_effort"] = reasoning_effort

        # Merge additional kwargs
        extra_kwargs.update(kwargs)

        # Use instructor to get structured output
        try:
            # Format model name for liteLLM based on provider
            formatted_kwargs = self._format_model_name(
                model=model,
                messages=messages,
                response_model=text_format,
                **extra_kwargs
            )
            response = self.instructor_client.chat.completions.create(**formatted_kwargs)
            return MockResponse(output_parsed=response)
        except Exception as e:
            # Re-raise with consistent error format
            raise e

    def chat_completions_create(self, **kwargs) -> MockChatCompletion:
        """
        Drop-in replacement for OpenAI's chat.completions.create() using liteLLM.

        Args:
            **kwargs: All parameters for chat completion

        Returns:
            MockChatCompletion object compatible with OpenAI's response structure
        """
        try:
            # Format model name for liteLLM based on provider
            kwargs = self._format_model_name(**kwargs)
            response = completion(**kwargs)
            content = response.choices[0].message.content
            return MockChatCompletion(content=content)
        except Exception as e:
            # Re-raise with consistent error format
            raise e

    async def achat_completions_create(self, **kwargs) -> MockChatCompletion:
        """
        Async version of chat.completions.create() using liteLLM.

        Args:
            **kwargs: All parameters for chat completion

        Returns:
            MockChatCompletion object compatible with OpenAI's response structure
        """
        try:
            # Format model name for liteLLM based on provider
            kwargs = self._format_model_name(**kwargs)
            response = await acompletion(**kwargs)
            content = response.choices[0].message.content
            return MockChatCompletion(content=content)
        except Exception as e:
            # Re-raise with consistent error format
            raise e

    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if an error should trigger a retry.

        Args:
            error: Exception that occurred

        Returns:
            True if the error warrants a retry
        """
        error_str = str(error).lower()
        return any(
            keyword in error_str
            for keyword in ["rate_limit", "429", "timeout", "connection", "server_error"]
        )

    def _calculate_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds

        Returns:
            Delay in seconds
        """
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, 1)
        return delay + jitter

    def responses_parse_with_retry(
        self,
        model: str,
        input: List[Dict[str, str]],
        text_format: Type[BaseModel],
        reasoning_effort: Optional[str] = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs
    ) -> MockResponse:
        """
        responses_parse with built-in retry logic.

        Args:
            model: Model name
            input: List of message dictionaries
            text_format: Pydantic model class for structured output
            reasoning_effort: Reasoning effort level (for compatible models)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            **kwargs: Additional parameters

        Returns:
            MockResponse with output_parsed attribute
        """
        for attempt in range(max_retries):
            try:
                return self.responses_parse(
                    model=model,
                    input=input,
                    text_format=text_format,
                    reasoning_effort=reasoning_effort,
                    **kwargs
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                if self._should_retry(e):
                    delay = self._calculate_delay(attempt, base_delay)
                    time.sleep(delay)
                else:
                    raise

    def chat_completions_create_with_retry(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs
    ) -> MockChatCompletion:
        """
        chat_completions_create with built-in retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            **kwargs: All parameters for chat completion

        Returns:
            MockChatCompletion object
        """
        for attempt in range(max_retries):
            try:
                return self.chat_completions_create(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                if self._should_retry(e):
                    delay = self._calculate_delay(attempt, base_delay)
                    time.sleep(delay)
                else:
                    raise

    async def achat_completions_create_with_retry(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        **kwargs
    ) -> MockChatCompletion:
        """
        Async chat_completions_create with built-in retry logic.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            **kwargs: All parameters for chat completion

        Returns:
            MockChatCompletion object
        """
        for attempt in range(max_retries):
            try:
                return await self.achat_completions_create(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                if self._should_retry(e):
                    delay = self._calculate_delay(attempt, base_delay)
                    await asyncio.sleep(delay)
                else:
                    raise


class ChatCompletion:
    """
    Namespace class to maintain OpenAI-style API structure.

    This allows code like `client.chat.completions.create()` to work
    with the unified client.
    """

    def __init__(self, client: UnifiedLLMClient):
        self.completions = ChatCompletions(client)


class ChatCompletions:
    """Completions namespace for chat operations."""

    def __init__(self, client: UnifiedLLMClient):
        self._client = client

    async def create(self, **kwargs):
        """Async chat completion creation - primary method for compatibility."""
        return await self._client.achat_completions_create(**kwargs)

    def create_sync(self, **kwargs):
        """Sync chat completion creation."""
        return self._client.chat_completions_create(**kwargs)


class Responses:
    """
    Namespace class for structured output parsing.

    This maintains compatibility with OpenAI's responses.parse() pattern.
    """

    def __init__(self, client: UnifiedLLMClient):
        self._client = client

    def parse(self, **kwargs):
        """Parse structured output using instructor."""
        return self._client.responses_parse(**kwargs)


class UnifiedLLMClientWithNamespaces(UnifiedLLMClient):
    """
    Extended unified client with OpenAI-style namespaced API.

    This version provides:
    - client.chat.completions.create()
    - client.responses.parse()

    While maintaining full liteLLM backend compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chat = ChatCompletion(self)
        self.responses = Responses(self)


# Factory functions for easy client creation

def create_sync_client(
    model: str,
    provider: str = "openai",
    **kwargs
) -> UnifiedLLMClientWithNamespaces:
    """
    Create a synchronous unified LLM client.

    Args:
        model: Model name
        provider: Provider name
        **kwargs: Additional configuration

    Returns:
        Configured UnifiedLLMClientWithNamespaces instance
    """
    return UnifiedLLMClientWithNamespaces(
        model=model,
        provider=provider,
        **kwargs
    )


def create_async_client(
    model: str,
    provider: str = "openai",
    **kwargs
) -> UnifiedLLMClientWithNamespaces:
    """
    Create an asynchronous unified LLM client.

    Note: The client itself is the same, but this factory function
    makes the intent clear and can be extended for async-specific
    configuration in the future.

    Args:
        model: Model name
        provider: Provider name
        **kwargs: Additional configuration

    Returns:
        Configured UnifiedLLMClientWithNamespaces instance
    """
    return UnifiedLLMClientWithNamespaces(
        model=model,
        provider=provider,
        **kwargs
    )