"""Main ConversationGenerator class for generating synthetic conversations."""

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable

# Handle imports for both module usage and CLI usage
try:
    from ..common.llm_client import create_async_client
except ImportError:
    # For CLI usage, add parent directory to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.llm_client import create_async_client

from .config import (
    ConversationConfig,
    get_language_name,
    random_config,
    select_agent_names,
)


@dataclass
class PromptPart:
    """Wraps config values with prompt text for template formatting."""

    value: Any
    prompt: str

    def __str__(self) -> str:
        return self.prompt


SYSTEM_PROMPT = """You are a synthetic conversation generator. Generate realistic customer support conversations.

Output a JSON object with exactly this structure:
{
  "subject": "Brief subject line for the conversation",
  "messages": [
    {
      "from_name": "Customer" or agent name from provided list (e.g., "Amanda S."),
      "from_type": "customer" or "agent",
      "message": "The message content"
    }
  ]
}

Rules:
- The first message must be from the customer
- For customer messages, use "Customer" in the from_name field
- For agent messages, use the actual agent name from the provided list in the from_name field (e.g., "Amanda S." instead of "Agent 1")
- If multiple agents are involved, each agent should consistently use their assigned name throughout the conversation
- Agent responses should be helpful and professional
- Messages should feel natural and conversational
- Use the specified language for ALL message content
- Match the specified channel style (email = formal, chat = casual)
- Include exactly the number of messages specified
- Only output valid JSON, no other text"""


def build_user_prompt(
    language: PromptPart,
    industry: PromptPart,
    problem: PromptPart,
    n_messages: PromptPart,
    n_agents: PromptPart,
    channel: PromptPart,
    agent_experience: PromptPart,
    agent_type: PromptPart,
    agent_names: PromptPart,
) -> str:
    """Build the user prompt from evaluated config parts."""
    return f"""Generate a customer support conversation with these parameters:

- Language: {language}
- Company: Klaus
- Industry: {industry}
- Customer problem: {problem}
- Number of messages: {n_messages}
- Number of agents: {n_agents}
- Agent names: {agent_names}
- Channel: {channel}
- Agent experience level: {agent_experience}
- Agent type: {agent_type}

Generate the conversation now."""


def evaluate_config(config: ConversationConfig) -> dict[str, PromptPart]:
    """Convert a config dict into PromptPart objects with descriptive prompts."""
    language_code = config.get("language", "en-us")
    language_name = get_language_name(language_code)

    # Select agent names based on n_agents
    n_agents = config.get("n_agents", 1)
    agent_names_list = select_agent_names(n_agents)
    agent_names_str = ", ".join(agent_names_list)

    return {
        "language": PromptPart(
            value=language_code,
            prompt=f"{language_name} ({language_code})",
        ),
        "industry": PromptPart(
            value=config.get("industry", "Retail"),
            prompt=config.get("industry", "Retail"),
        ),
        "problem": PromptPart(
            value=config.get("problem", "General inquiry"),
            prompt=config.get("problem", "General inquiry"),
        ),
        "n_messages": PromptPart(
            value=config.get("n_messages", 6),
            prompt=str(config.get("n_messages", 6)),
        ),
        "n_agents": PromptPart(
            value=n_agents,
            prompt=f"{n_agents} agent(s)",
        ),
        "agent_names": PromptPart(
            value=agent_names_list,
            prompt=agent_names_str,
        ),
        "channel": PromptPart(
            value=config.get("channel", "chat"),
            prompt=config.get("channel", "chat"),
        ),
        "agent_experience": PromptPart(
            value=config.get("agent_experience", "senior"),
            prompt=config.get("agent_experience", "senior"),
        ),
        "agent_type": PromptPart(
            value=config.get("agent_type", "human"),
            prompt=config.get("agent_type", "human"),
        ),
    }


class ConversationGenerator:
    """Generates synthetic customer support conversations using liteLLM."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
    ):
        """Initialize the generator.

        Args:
            model: Model to use for generation.
            api_key: API key. If not provided, uses OPENAI_API_KEY env var.
            base_url: API base URL. If not provided, uses OPENAI_BASE_URL
                     env var or defaults to EU endpoint.
            provider: Provider name (openai, groq, etc.).
        """
        self.model = model
        self.provider = provider

        # Default to EU endpoint for OpenAI
        if base_url is None and provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://eu.api.openai.com/v1")

        self.client = create_async_client(
            model=model,
            provider=provider,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    async def generate(
        self,
        config: ConversationConfig | None = None,
        n_agents: int | None = None,
    ) -> dict[str, Any]:
        """Generate a single conversation.

        Args:
            config: Configuration for the conversation. If None, uses random config.
            n_agents: Override number of agents (optional).

        Returns:
            Generated conversation with subject, messages, and metadata.
        """
        if config is None:
            config = random_config()

        if n_agents is not None:
            config = {**config, "n_agents": n_agents}

        evaluated = evaluate_config(config)
        user_prompt = build_user_prompt(
            language=evaluated["language"],
            industry=evaluated["industry"],
            problem=evaluated["problem"],
            n_messages=evaluated["n_messages"],
            n_agents=evaluated["n_agents"],
            agent_names=evaluated["agent_names"],
            channel=evaluated["channel"],
            agent_experience=evaluated["agent_experience"],
            agent_type=evaluated["agent_type"],
        )

        # Retry with exponential backoff
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                else:
                    raise

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from OpenAI")

        result = json.loads(content)

        # Add model at top level for easy parsing
        result["model"] = self.model

        # Add metadata
        result["_metadata"] = {
            "language": evaluated["language"].value,
            "industry": evaluated["industry"].value,
            "problem": evaluated["problem"].value,
            "n_messages": evaluated["n_messages"].value,
            "n_agents": evaluated["n_agents"].value,
            "channel": evaluated["channel"].value,
            "agent_experience": evaluated["agent_experience"].value,
            "agent_type": evaluated["agent_type"].value,
            "model": self.model,
        }

        return result

    async def generate_batch(
        self,
        n: int,
        config_fn: Callable[[], ConversationConfig] | None = None,
        parallelization: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate multiple conversations in parallel.

        Args:
            n: Number of conversations to generate.
            config_fn: Function that returns a config for each conversation.
                      If None, uses random_config().
            parallelization: Maximum number of concurrent requests.

        Returns:
            List of generated conversations.
        """
        if config_fn is None:
            config_fn = random_config

        semaphore = asyncio.Semaphore(parallelization)

        async def generate_with_semaphore(idx: int) -> dict[str, Any]:
            async with semaphore:
                config = config_fn()
                return await self.generate(config=config)

        tasks = [generate_with_semaphore(i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        return list(results)
