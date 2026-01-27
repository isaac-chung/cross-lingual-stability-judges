"""CLI entry point for conversation parser."""

import argparse
import glob
import json
import sys
from pathlib import Path

from .parser import combine_files, compute_stats, get_language, group_by_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine and validate JSONL files from conversation generator",
        prog="python -m conversation_parser",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more JSONL files or glob patterns",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/combined_{lang}.json)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics to stderr",
    )
    return parser.parse_args()


def expand_globs(patterns: list[str]) -> list[Path]:
    """Expand glob patterns to list of paths."""
    paths = []
    for pattern in patterns:
        # Try glob expansion
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            # Treat as literal path if no glob matches
            path = Path(pattern)
            if path.exists():
                paths.append(path)
            else:
                print(f"Warning: No matches for '{pattern}'", file=sys.stderr)
    return sorted(set(paths))  # Deduplicate and sort


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Expand glob patterns
    paths = expand_globs(args.inputs)
    if not paths:
        print("Error: No input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(paths)} file(s)...", file=sys.stderr)

    # Combine files
    conversations = combine_files(paths)

    if not conversations:
        print("Error: No conversations found in input files", file=sys.stderr)
        sys.exit(1)

    # Print stats if requested
    if args.stats:
        stats = compute_stats(conversations)
        print("\n--- Statistics ---", file=sys.stderr)
        print(f"Total conversations: {stats['total']}", file=sys.stderr)
        print("\nBy model:", file=sys.stderr)
        for model, count in sorted(stats["by_model"].items()):
            print(f"  {model}: {count}", file=sys.stderr)
        print("\nBy language:", file=sys.stderr)
        for lang, count in sorted(stats["by_language"].items()):
            print(f"  {lang}: {count}", file=sys.stderr)
        print("\nBy model and language:", file=sys.stderr)
        for model, langs in sorted(stats["by_model_language"].items()):
            print(f"  {model}:", file=sys.stderr)
            for lang, count in sorted(langs.items()):
                print(f"    {lang}: {count}", file=sys.stderr)
        print("", file=sys.stderr)

    # Write output if path specified (or default)
    if args.output is not None or not args.stats:
        if args.output is None:
            # Determine language for filename
            languages = set(get_language(conv) for conv in conversations)
            lang_for_filename = languages.pop() if len(languages) == 1 else "mixed"
            output_path = f"data/combined_{lang_for_filename}.json"
        else:
            output_path = args.output

        # Group by model for output
        grouped = group_by_model(conversations)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grouped, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"Wrote {len(conversations)} conversation(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
