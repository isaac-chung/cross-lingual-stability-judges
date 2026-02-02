#!/usr/bin/env python3
"""
CLI entry point for HuggingFace dataset loader.

Usage:
    python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini
    python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini -o data/combined.json
    python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini --languages et,fi
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .loader import HFLoader


console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

    # Reduce noise from external libraries
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load HuggingFace dataset and convert to analysis format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load dataset and save to default location
  python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini

  # Specify output path
  python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini -o data/combined.json

  # Load only specific languages
  python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini --languages et,fi

  # Limit number of conversations (useful for testing)
  python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini --max-conversations 100

  # Preview without saving
  python -m hf_loader isaacchung/controlled-generated-convos-gpt-4.1-mini --dry-run
        """
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="HuggingFace dataset name (e.g., 'isaacchung/controlled-generated-convos-gpt-4.1-mini')"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file path (default: data/combined_hf_{dataset_suffix}.json)"
    )

    parser.add_argument(
        "--languages",
        type=str,
        help="Comma-separated language codes to include (e.g., 'et,fi,hu')"
    )

    parser.add_argument(
        "--max-conversations",
        type=int,
        help="Maximum conversations per language (useful for testing)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview dataset info without saving"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug output"
    )

    return parser.parse_args()


def parse_languages(language_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated language string."""
    if not language_str:
        return None
    return [lang.strip() for lang in language_str.split(",") if lang.strip()]


def get_default_output_path(dataset_name: str) -> Path:
    """Generate default output path from dataset name."""
    # Extract suffix from dataset name (e.g., 'gpt-4.1-mini' from full path)
    suffix = dataset_name.split("/")[-1].replace("controlled-generated-convos-", "")
    return Path(f"data/combined_hf_{suffix}.json")


def display_stats(stats: dict, dataset_name: str, output_path: Optional[Path] = None) -> None:
    """Display statistics in a formatted table."""
    table = Table(title=f"Dataset: {dataset_name}")

    table.add_column("Model", style="bold blue")
    table.add_column("Conversations", justify="right", style="green")

    total = 0
    for model, count in stats.items():
        table.add_row(model, f"{count:,}")
        total += count

    table.add_row("Total", f"{total:,}", style="bold")

    console.print(table)

    if output_path:
        console.print(f"\n[green]Saved to:[/green] {output_path}")


def main() -> None:
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    args = parse_arguments()

    setup_logging(args.verbose)

    # Get HF token from environment (optional, for private datasets)
    token = os.getenv("HF_TOKEN")

    # Parse languages
    languages = parse_languages(args.languages)

    # Determine output path
    output_path = Path(args.output) if args.output else get_default_output_path(args.dataset)

    # Create loader
    loader = HFLoader(args.dataset, token=token)

    if args.dry_run:
        console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
        if languages:
            console.print(f"[bold]Languages:[/bold] {', '.join(languages)}")

        # Load and display stats without saving
        data = loader.load(languages=languages, max_conversations=args.max_conversations)
        stats = {model: len(convs) for model, convs in data.items()}
        display_stats(stats, args.dataset)
        console.print("\n[dim]Dry run - no file saved. Remove --dry-run to save.[/dim]")
    else:
        console.print(f"[bold]Loading dataset:[/bold] {args.dataset}")
        if languages:
            console.print(f"[bold]Languages:[/bold] {', '.join(languages)}")
        console.print(f"[bold]Output:[/bold] {output_path}\n")

        # Load and save
        stats = loader.save(
            output_path=output_path,
            languages=languages,
            max_conversations=args.max_conversations
        )
        display_stats(stats, args.dataset, output_path)

        console.print("\n[green]✅ Dataset loaded and converted successfully![/green]")
        console.print(f"\n[dim]You can now run analysis with:[/dim]")
        console.print(f"  python -m llm_judge {output_path} --stats")
        console.print(f"  python -m label_recovery {output_path} --stats")


if __name__ == "__main__":
    main()
