"""CLI entry point for judge ablation analysis."""

import argparse
import glob
import sys

from .analyzer import JudgeAblationAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare label recovery results from different judge models",
        prog="python -m judge_ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare results from different judge models
  python -m judge_ablation data/label_recovery_*.jsonl \\
      --ground-truth data/combined_et.json

  # Save detailed results to JSON
  python -m judge_ablation data/label_recovery_*.jsonl \\
      --ground-truth config.json -o results.json

  # Load and display saved analysis results
  python -m judge_ablation --from-results results.json
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSONL files to compare (supports glob patterns)",
    )
    parser.add_argument(
        "--ground-truth",
        metavar="FILE",
        help="Ground truth JSON file (required unless using --from-results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional JSON output path for detailed results",
    )
    parser.add_argument(
        "--from-results",
        type=str,
        metavar="FILE",
        help="Load and display saved analysis results (no new analysis)",
    )
    return parser.parse_args()


def expand_glob_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns in file list.

    Args:
        patterns: List of file paths or glob patterns.

    Returns:
        List of expanded file paths.
    """
    expanded = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(matches)
        else:
            # Keep non-matching patterns (will be handled as missing files)
            expanded.append(pattern)
    return expanded


def main() -> None:
    """Main entry point."""
    args = parse_args()
    analyzer = JudgeAblationAnalyzer()

    # Handle --from-results mode (display saved analysis)
    if args.from_results:
        analyzer.display_saved_results(args.from_results)
        return

    # Validate arguments for new analysis
    if not args.files:
        print("Error: No files specified", file=sys.stderr)
        sys.exit(1)

    if not args.ground_truth:
        print("Error: --ground-truth is required", file=sys.stderr)
        sys.exit(1)

    # Expand glob patterns
    files = expand_glob_patterns(args.files)
    if not files:
        print("Error: No files matched", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    result = analyzer.run_analysis(
        result_paths=files,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
    )

    if not result:
        sys.exit(1)


if __name__ == "__main__":
    main()
