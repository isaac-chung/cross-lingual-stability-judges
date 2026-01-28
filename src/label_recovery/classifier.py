"""Core classification logic for Label Recovery."""

import json
import os
import random
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .models import ClassificationResult, ConversationClassification
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class LabelRecoveryClassifier:
    """LLM-based conversation classifier for label recovery."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        provider: str = "openai",
        reasoning_effort: str | None = None,
    ):
        """Initialize the classifier.

        Args:
            model: Model to use for classification.
            provider: API provider ("openai" or "groq").
            reasoning_effort: Reasoning effort for o-series models.
        """
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.client = self._create_client()

    def _get_judge_model_name(self) -> str:
        """Get judge model name including reasoning effort if applicable.

        Returns:
            Model name, with reasoning effort suffix for o-series models.
        """
        if self.reasoning_effort and self.model.startswith("o"):
            return f"{self.model}-{self.reasoning_effort}"
        return self.model

    def _create_client(self) -> OpenAI:
        """Create client based on provider."""
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                msg = "GROQ_API_KEY environment variable required for Groq provider"
                raise ValueError(msg)
            return OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            # OpenAI provider (default)
            base_url = os.getenv("OPENAI_BASE_URL", "https://eu.api.openai.com/v1")
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
            )

    def format_conversation(self, conv: dict[str, Any]) -> str:
        """Format conversation for prompt.

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

    def _classify_with_openai(
        self,
        messages: list[dict[str, str]],
        conversation_id: int,
        generator_model: str,
        start_time: float,
        source_file: str | None = None,
    ) -> ClassificationResult:
        """Classify using OpenAI's responses.parse API.

        Args:
            messages: List of message dicts for the API.
            conversation_id: Unique identifier for the conversation.
            generator_model: Model that generated the conversation.
            start_time: Start time for processing time calculation.
            source_file: Source file path.

        Returns:
            ClassificationResult with classification labels and metadata.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text_format": ConversationClassification,
        }

        # Add reasoning effort for o-series models
        if self.reasoning_effort and self.model.startswith("o"):
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

        parsed_result = response.output_parsed
        if parsed_result is None:
            raise ValueError("Failed to parse classification response")

        return ClassificationResult(
            conversation_id=conversation_id,
            generator_model=generator_model,
            judge_model=self._get_judge_model_name(),
            industry=parsed_result.industry,
            problem=parsed_result.problem,
            channel=parsed_result.channel,
            agent_experience=parsed_result.agent_experience,
            agent_type=parsed_result.agent_type,
            explanation=parsed_result.explanation,
            processing_time=time.time() - start_time,
            source_file=source_file,
        )

    def _classify_with_groq(
        self,
        messages: list[dict[str, str]],
        conversation_id: int,
        generator_model: str,
        start_time: float,
        source_file: str | None = None,
    ) -> ClassificationResult:
        """Classify using Groq's chat completions API with JSON mode.

        Args:
            messages: List of message dicts for the API.
            conversation_id: Unique identifier for the conversation.
            generator_model: Model that generated the conversation.
            start_time: Start time for processing time calculation.
            source_file: Source file path.

        Returns:
            ClassificationResult with classification labels and metadata.
        """
        json_schema_instruction = """

IMPORTANT: You must respond with a valid JSON object containing these exact fields:
{
    "industry": "<industry from the list>",
    "problem": "<problem type from the list>",
    "channel": "<email or chat>",
    "agent_experience": "<junior or senior>",
    "agent_type": "<human or bot>",
    "explanation": "<brief justification for classifications>"
}

Respond ONLY with the JSON object, no additional text."""

        # Modify the system message to include JSON instruction
        modified_messages = messages.copy()
        modified_messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + json_schema_instruction,
        }

        # Use Groq's chat completions API with JSON mode
        response = self.client.chat.completions.create(
            model=self.model,
            messages=modified_messages,
            response_format={"type": "json_object"},
            temperature=0.7,
            top_p=0.8,
        )

        # Parse JSON response
        raw_response = response.choices[0].message.content
        parsed_json = json.loads(raw_response)

        # Validate with Pydantic model
        parsed_result = ConversationClassification(**parsed_json)

        return ClassificationResult(
            conversation_id=conversation_id,
            generator_model=generator_model,
            judge_model=self._get_judge_model_name(),
            industry=parsed_result.industry,
            problem=parsed_result.problem,
            channel=parsed_result.channel,
            agent_experience=parsed_result.agent_experience,
            agent_type=parsed_result.agent_type,
            explanation=parsed_result.explanation,
            processing_time=time.time() - start_time,
            source_file=source_file,
        )

    def classify(
        self,
        conv: dict[str, Any],
        conversation_id: int = 0,
        generator_model: str = "unknown",
        source_file: str | None = None,
    ) -> ClassificationResult:
        """Classify a single conversation.

        Args:
            conv: Conversation dict with messages.
            conversation_id: Unique identifier for the conversation.
            generator_model: Model that generated the conversation.
            source_file: Source file path.

        Returns:
            ClassificationResult with classification labels and metadata.
        """
        start_time = time.time()

        try:
            user_prompt = self.format_conversation(conv)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            if self.provider == "groq":
                return self._classify_with_groq(
                    messages, conversation_id, generator_model, start_time, source_file
                )
            else:
                return self._classify_with_openai(
                    messages, conversation_id, generator_model, start_time, source_file
                )

        except Exception as e:
            return ClassificationResult(
                conversation_id=conversation_id,
                generator_model=generator_model,
                judge_model=self._get_judge_model_name(),
                error=str(e),
                processing_time=time.time() - start_time,
                source_file=source_file,
            )

    def classify_batch(
        self,
        convs: list[tuple[str, int, dict[str, Any]]],
        workers: int = 5,
        on_complete: Callable[[int, int], None] | None = None,
        source_file: str | None = None,
    ) -> list[ClassificationResult]:
        """Concurrent classification with ThreadPoolExecutor.

        Args:
            convs: List of (generator_model, conv_idx, conv) tuples.
            workers: Number of concurrent workers.
            on_complete: Callback called after each classification.
            source_file: Source file path.

        Returns:
            List of ClassificationResult results in the same order as input.
        """
        results: list[ClassificationResult | None] = [None] * len(convs)

        def classify_single(
            args: tuple[int, str, int, dict[str, Any]],
        ) -> tuple[int, ClassificationResult]:
            list_idx, generator_model, conv_idx, conv = args
            result = self.classify(
                conv,
                conversation_id=conv_idx,
                generator_model=generator_model,
                source_file=source_file,
            )
            return list_idx, result

        # Prepare arguments
        args_list = [
            (i, gen_model, conv_idx, conv)
            for i, (gen_model, conv_idx, conv) in enumerate(convs)
        ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(classify_single, args): args[0] for args in args_list
            }

            completed = 0
            for future in as_completed(future_to_idx):
                list_idx, result = future.result()
                results[list_idx] = result
                completed += 1
                if on_complete:
                    on_complete(completed, len(convs))

        return results  # type: ignore

    def analyze_results(self, results: list[ClassificationResult]) -> dict[str, Any]:
        """Analyze classification results and generate statistics.

        Args:
            results: List of ClassificationResult objects.

        Returns:
            Dictionary with analysis results per category.
        """
        successful = [r for r in results if r.is_successful()]
        failed = [r for r in results if not r.is_successful()]

        if not successful:
            return {
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
                "success_rate": 0.0,
            }

        analysis = {
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results),
            "industry_distribution": dict(
                Counter(r.industry for r in successful)
            ),
            "problem_distribution": dict(Counter(r.problem for r in successful)),
            "channel_distribution": dict(Counter(r.channel for r in successful)),
            "agent_experience_distribution": dict(
                Counter(r.agent_experience for r in successful)
            ),
            "agent_type_distribution": dict(
                Counter(r.agent_type for r in successful)
            ),
        }

        # Processing time stats
        processing_times = [r.processing_time for r in successful if r.processing_time]
        if processing_times:
            analysis["processing"] = {
                "avg_seconds": sum(processing_times) / len(processing_times),
                "total_seconds": sum(processing_times),
            }

        return analysis

    def display_results_table(
        self,
        results_by_model: dict[str, list[ClassificationResult]],
    ) -> None:
        """Display Rich table comparing classification metrics across models.

        Args:
            results_by_model: Dict mapping generator model names to results.
        """
        console = Console()
        table = Table(title="Label Recovery Classification Results by Generator Model")

        table.add_column("Model", justify="left", style="cyan", no_wrap=True)
        table.add_column("Success Rate", justify="center", style="yellow")
        table.add_column("Total", justify="center", style="blue")
        table.add_column("Top Industry", justify="center", style="magenta")
        table.add_column("Top Problem", justify="center", style="green")
        table.add_column("Top Channel", justify="center", style="red")

        for model_name, results in results_by_model.items():
            successful = [r for r in results if r.is_successful()]

            if successful:
                industries = Counter(r.industry for r in successful)
                problems = Counter(r.problem for r in successful)
                channels = Counter(r.channel for r in successful)

                top_industry = industries.most_common(1)[0][0] if industries else "N/A"
                top_problem = problems.most_common(1)[0][0] if problems else "N/A"
                top_channel = channels.most_common(1)[0][0] if channels else "N/A"

                success_rate = len(successful) / len(results) * 100

                # Shorten model name for display
                display_name = model_name.replace("anthropic.", "").replace("cohere.", "")
                display_name = display_name.replace("meta.", "").replace("mistral.", "")

                table.add_row(
                    display_name,
                    f"{success_rate:.1f}%",
                    str(len(results)),
                    top_industry[:25] + "..." if len(top_industry) > 25 else top_industry,
                    top_problem,
                    top_channel,
                )
            else:
                display_name = model_name.replace("anthropic.", "").replace("cohere.", "")
                display_name = display_name.replace("meta.", "").replace("mistral.", "")
                table.add_row(display_name, "0.0%", str(len(results)), "N/A", "N/A", "N/A")

        console.print(table)
