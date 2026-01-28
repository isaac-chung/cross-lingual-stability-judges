#!/usr/bin/env python3
"""
CLI entry point for HuggingFace dataset upload module.

Usage:
    python -m hf_uploader --model gpt-4.1-mini
    python -m hf_uploader --files "data/convo_*.jsonl"
    python -m hf_uploader --model gpt-4.1-mini --languages et,fi --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional
import glob

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from .config import HFConfig, SUPPORTED_LANGUAGES
from .uploader import HFUploader


# Set up rich console
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: Enable debug logging if True
    """
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
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Upload conversation datasets to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all conversations for a specific model
  python -m hf_uploader --model gpt-4.1-mini

  # Upload specific files using glob pattern
  python -m hf_uploader --files "data/convo_gpt-4.1_*.jsonl"

  # Upload with language filter and preview first
  python -m hf_uploader --model gpt-4.1-mini --languages et,fi --dry-run

  # Create private dataset with custom description
  python -m hf_uploader --model gpt-4.1-mini --private --description "My custom dataset"

  # Add new conversations to existing dataset (default behavior)
  python -m hf_uploader --model gpt-4.1-mini

  # Replace existing dataset entirely instead of merging
  python -m hf_uploader --model gpt-4.1-mini --overwrite

Environment variables required:
  HF_TOKEN      - HuggingFace API token
  HF_USERNAME   - HuggingFace username
        """
    )

    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--files",
        type=str,
        help="Glob pattern for conversation files to upload (e.g., 'data/convo_*.jsonl')"
    )
    input_group.add_argument(
        "--model",
        type=str,
        help="Upload all conversations for this model name (e.g., 'gpt-4.1-mini')"
    )

    # Filtering options
    parser.add_argument(
        "--languages",
        type=str,
        help=f"Comma-separated language codes to include (e.g., 'et,fi'). "
             f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
    )

    # Dataset options
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: public)"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Custom description for the dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Override the auto-generated dataset name"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing dataset entirely instead of merging new conversations"
    )

    # Operation modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without actually uploading"
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate language codes
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(",")]
        invalid_languages = [lang for lang in languages if lang not in SUPPORTED_LANGUAGES]
        if invalid_languages:
            raise ValueError(
                f"Unsupported languages: {invalid_languages}. "
                f"Supported: {SUPPORTED_LANGUAGES}"
            )

    # Check for conflicting quiet/verbose
    if args.quiet and args.verbose:
        raise ValueError("Cannot use --quiet and --verbose together")


def resolve_file_paths(pattern: str) -> List[Path]:
    """Resolve glob pattern to file paths.

    Args:
        pattern: Glob pattern for files

    Returns:
        List of resolved file paths

    Raises:
        ValueError: If no files match pattern
    """
    file_paths = []
    for path_str in glob.glob(pattern):
        path = Path(path_str)
        if path.exists() and path.is_file():
            file_paths.append(path)

    if not file_paths:
        raise ValueError(f"No files found matching pattern: {pattern}")

    return sorted(file_paths)


def parse_languages(language_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated language string.

    Args:
        language_str: Comma-separated language codes

    Returns:
        List of language codes or None
    """
    if not language_str:
        return None

    languages = [lang.strip() for lang in language_str.split(",")]
    return [lang for lang in languages if lang]  # Filter out empty strings


def display_results(results: dict[str, str], dry_run: bool = False) -> None:
    """Display upload results in a formatted table.

    Args:
        results: Dictionary mapping models/datasets to URLs
        dry_run: Whether this was a dry run
    """
    if not results:
        console.print("[yellow]No results to display[/yellow]")
        return

    table = Table(
        title="🚀 Upload Results" if not dry_run else "👀 Dry Run Preview",
        title_style="bold cyan"
    )

    table.add_column("Model/Dataset", style="bold blue")
    table.add_column("Status", style="green" if not dry_run else "yellow")
    table.add_column("URL", style="dim")

    for model, url in results.items():
        status = "✅ Uploaded" if not dry_run else "🔍 Would upload"
        table.add_row(model, status, url)

    console.print(table)


def display_error(error: Exception) -> None:
    """Display error message in a formatted panel.

    Args:
        error: Exception that occurred
    """
    error_panel = Panel(
        f"[red]{type(error).__name__}[/red]: {str(error)}",
        title="❌ Error",
        title_align="left",
        border_style="red"
    )
    console.print(error_panel)


def display_welcome() -> None:
    """Display welcome message with environment check."""
    welcome_text = Text()
    welcome_text.append("🤗 HuggingFace Dataset Uploader\n", style="bold cyan")
    welcome_text.append("Upload conversation datasets to HuggingFace Hub", style="dim")

    welcome_panel = Panel(
        welcome_text,
        title="Welcome",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(welcome_panel)


async def main() -> None:
    """Main entry point for the CLI application."""
    try:
        # Parse arguments
        args = parse_arguments()
        validate_arguments(args)

        # Set up logging
        if not args.quiet:
            setup_logging(args.verbose)
            display_welcome()

        # Create configuration
        try:
            config = HFConfig.from_environment(
                private=args.private,
                overwrite=args.overwrite,
                dataset_name_override=args.dataset_name,
                custom_description=args.description
            )
        except ValueError as e:
            display_error(e)
            console.print("\n[dim]Make sure to set HF_TOKEN and HF_USERNAME environment variables.[/dim]")
            sys.exit(1)

        # Create uploader
        try:
            # Skip credential validation for dry-run mode
            validate_credentials = not args.dry_run
            uploader = HFUploader(config, validate_credentials=validate_credentials)
        except ValueError as e:
            display_error(e)
            console.print("\n[dim]Check your HuggingFace credentials and try again.[/dim]")
            sys.exit(1)

        # Process based on input type
        results = {}

        if args.model:
            # Upload by model
            languages = parse_languages(args.languages)

            if not args.quiet:
                console.print(f"\n[bold]Uploading conversations for model:[/bold] {args.model}")
                if languages:
                    console.print(f"[bold]Languages:[/bold] {', '.join(languages)}")

            results = await uploader.upload_by_model(
                model=args.model,
                languages=languages,
                dry_run=args.dry_run
            )

        elif args.files:
            # Upload specific files
            try:
                file_paths = resolve_file_paths(args.files)
            except ValueError as e:
                display_error(e)
                sys.exit(1)

            if not args.quiet:
                console.print(f"\n[bold]Uploading {len(file_paths)} files[/bold]")

            results = await uploader.upload_files(
                file_paths=file_paths,
                dry_run=args.dry_run
            )

        # Display results
        if not args.quiet:
            display_results(results, args.dry_run)

        if args.dry_run:
            console.print("\n[dim]This was a dry run. Use without --dry-run to actually upload.[/dim]")
        else:
            console.print("\n[green]✅ Upload completed successfully![/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Upload cancelled by user[/yellow]")
        sys.exit(130)

    except Exception as e:
        if not args.quiet:
            display_error(e)
        else:
            console.print(f"Error: {e}", style="red")

        if args.verbose:
            console.print_exception()

        sys.exit(1)


if __name__ == "__main__":
    # Ensure we're running on a supported Python version
    if sys.version_info < (3, 8):
        console.print("[red]Error: Python 3.8+ required[/red]")
        sys.exit(1)

    asyncio.run(main())