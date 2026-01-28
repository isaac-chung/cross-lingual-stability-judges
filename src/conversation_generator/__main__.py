"""CLI entry point for conversation generator."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .config import LANGUAGES, random_config
from .generator import ConversationGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer support conversations",
        prog="python -m conversation_generator",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of conversations to generate (default: 1)",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=[code for code, _ in LANGUAGES],
        default=None,
        help="Language code: et, fi, hu, en-us (default: random)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/{model}_{lang}_{datetime}.jsonl)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=5,
        help="Parallelization level for batch generation (default: 5)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://eu.api.openai.com/v1",
        help="OpenAI API base URL (default: https://eu.api.openai.com/v1)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (default: openai, options: openai, groq)",
    )
    return parser.parse_args()


def make_config_fn(language: str | None):
    """Create a config function that optionally fixes the language."""

    def config_fn():
        config = random_config()
        if language is not None:
            config["language"] = language  # type: ignore
        return config

    return config_fn


async def main() -> None:
    """Main entry point."""
    load_dotenv()
    args = parse_args()

    generator = ConversationGenerator(
        model=args.model,
        base_url=args.base_url,
        provider=args.provider,
    )

    config_fn = make_config_fn(args.language)

    # Determine language for filename (use "mixed" if random)
    lang_for_filename = args.language if args.language else "mixed"

    # Build output path with model, language, and datetime if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"data/{args.model}_{lang_for_filename}_{timestamp}.jsonl"
    else:
        output_path = args.output

    if args.count == 1:
        # Single generation
        result = await generator.generate(config=config_fn())
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        # Batch generation
        results = await generator.generate_batch(
            n=args.count,
            config_fn=config_fn,
            parallelization=args.parallel,
        )
        # JSONL format for batches
        output = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in results
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output + "\n", encoding="utf-8")
    print(f"Wrote {args.count} conversation(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
