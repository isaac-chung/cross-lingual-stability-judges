"""CLI entry point for Label Recovery classification."""

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .classifier import LabelRecoveryClassifier
from .evaluator import (
    display_detailed_results,
    display_results_table,
    evaluate_predictions,
    load_ground_truth,
)
from .models import ClassificationResult


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Label Recovery classification for synthetic conversations",
        prog="python -m label_recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new classification
  python -m label_recovery data/combined_et.json --stats

  # Load existing JSONL results (no classification)
  python -m label_recovery --from-results data/label_recovery_*.jsonl

  # Evaluate predictions against ground truth
  python -m label_recovery --evaluate data/*.jsonl --ground-truth config.json
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Combined JSON file (default: auto-detect from data/combined_*.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/label_recovery_{lang}_{datetime}.jsonl)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Classifier model (default: gpt-5-mini)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=5,
        help="Concurrent workers (default: 5)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for o-series models",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "groq"],
        default="openai",
        help="API provider (default: openai)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics after classification",
    )
    parser.add_argument(
        "--from-results",
        nargs="+",
        metavar="FILE",
        help="Load existing JSONL results and display analysis (no classification)",
    )
    parser.add_argument(
        "--evaluate",
        nargs="+",
        metavar="FILE",
        help="Evaluate JSONL results against ground truth",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        metavar="FILE",
        help="Ground truth JSON file (required with --evaluate)",
    )
    return parser.parse_args()


def auto_detect_input() -> Path | None:
    """Auto-detect input file from data/combined_*.json."""
    matches = glob.glob("data/combined_*.json")
    if len(matches) == 1:
        return Path(matches[0])
    elif len(matches) > 1:
        print(f"Multiple combined files found: {matches}", file=sys.stderr)
        print("Please specify one explicitly", file=sys.stderr)
        return None
    return None


def extract_language(input_path: Path) -> str:
    """Extract language code from input filename."""
    stem = input_path.stem
    if stem.startswith("combined_"):
        return stem[9:]  # Remove "combined_" prefix
    return "unknown"


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in filename.

    Args:
        model: Model name (may contain slashes).

    Returns:
        Sanitized model name with slashes replaced by dashes.
    """
    return model.replace("/", "-")


def get_judge_model_filename(model: str, reasoning_effort: str | None) -> str:
    """Get judge model name for filename.

    Args:
        model: Model name.
        reasoning_effort: Reasoning effort for o-series models.

    Returns:
        Model name with reasoning effort suffix for o-series models.
    """
    sanitized = sanitize_model_name(model)
    if reasoning_effort and model.startswith("o"):
        return f"{sanitized}-{reasoning_effort}"
    return sanitized


def load_results_from_jsonl(paths: list[str]) -> list[ClassificationResult]:
    """Load classification results from JSONL files.

    Args:
        paths: List of file paths (supports glob patterns).

    Returns:
        List of ClassificationResult objects.
    """
    results = []

    # Expand glob patterns
    expanded_paths = []
    for pattern in paths:
        matches = glob.glob(pattern)
        if matches:
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(pattern)

    for path in expanded_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Warning: File not found: {path}", file=sys.stderr)
            continue
        with open(path_obj, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    results.append(ClassificationResult.from_dict(data))

    return results


def print_stats(results: list[ClassificationResult]) -> None:
    """Print summary statistics for classification results."""
    print("\n--- Classification Statistics ---", file=sys.stderr)
    print(f"Total classified: {len(results)}", file=sys.stderr)

    successful = [r for r in results if r.is_successful()]
    failed = [r for r in results if not r.is_successful()]

    print(f"Successful: {len(successful)}", file=sys.stderr)
    print(f"Failed: {len(failed)}", file=sys.stderr)
    print(f"Success rate: {100 * len(successful) / len(results):.1f}%", file=sys.stderr)

    if failed:
        print("\nErrors:", file=sys.stderr)
        error_counts: dict[str, int] = {}
        for r in failed:
            err = r.error or "Incomplete classification"
            error_counts[err] = error_counts.get(err, 0) + 1
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {error}: {count}", file=sys.stderr)


def display_from_results_stats(results: list[ClassificationResult]) -> None:
    """Display statistics from loaded results."""
    # Group by generator model
    by_model: dict[str, list[ClassificationResult]] = {}
    for r in results:
        model = r.generator_model
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    # Create classifier just for display method
    classifier = LabelRecoveryClassifier.__new__(LabelRecoveryClassifier)
    classifier.display_results_table(by_model)

    # Print detailed stats per model
    print("\n--- Detailed Statistics by Generator Model ---", file=sys.stderr)
    for model in sorted(by_model.keys()):
        model_results = by_model[model]
        successful = [r for r in model_results if r.is_successful()]
        n = len(model_results)
        success_rate = len(successful) / n * 100 if n > 0 else 0

        print(f"\n  {model} (n={n}, {success_rate:.1f}% success):", file=sys.stderr)

        if successful:
            # Show top categories
            from collections import Counter

            industries = Counter(r.industry for r in successful)
            problems = Counter(r.problem for r in successful)
            channels = Counter(r.channel for r in successful)
            experiences = Counter(r.agent_experience for r in successful)
            types = Counter(r.agent_type for r in successful)

            top_industry = industries.most_common(3)
            top_problem = problems.most_common(3)

            print(f"    Top industries: {top_industry}", file=sys.stderr)
            print(f"    Top problems: {top_problem}", file=sys.stderr)
            print(f"    Channels: {dict(channels)}", file=sys.stderr)
            print(f"    Experience: {dict(experiences)}", file=sys.stderr)
            print(f"    Agent type: {dict(types)}", file=sys.stderr)


def run_evaluation(
    result_paths: list[str],
    ground_truth_path: str,
) -> None:
    """Run evaluation of predictions against ground truth.

    Args:
        result_paths: Paths to JSONL result files.
        ground_truth_path: Path to ground truth JSON file.
    """
    # Load results
    results = load_results_from_jsonl(result_paths)
    if not results:
        print("Error: No results loaded from specified files", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(results)} results", file=sys.stderr)

    # Load ground truth
    ground_truth = load_ground_truth([ground_truth_path])
    if not ground_truth:
        print("Error: No ground truth loaded", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(ground_truth)} ground truth entries", file=sys.stderr)

    # Group results by generator model
    by_model: dict[str, list[ClassificationResult]] = {}
    for r in results:
        model = r.generator_model
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    # Evaluate each model
    all_eval_results = []
    for model_name, model_results in by_model.items():
        eval_result = evaluate_predictions(model_results, ground_truth)
        eval_result["model_name"] = model_name
        all_eval_results.append(eval_result)

    # Display results
    display_results_table(all_eval_results)
    display_detailed_results(all_eval_results)


def main() -> None:
    """Main entry point."""
    load_dotenv()
    args = parse_args()

    # Handle --evaluate mode
    if args.evaluate:
        if not args.ground_truth:
            print("Error: --ground-truth is required with --evaluate", file=sys.stderr)
            sys.exit(1)
        run_evaluation(args.evaluate, args.ground_truth)
        return

    # Handle --from-results mode (analysis only, no classification)
    if args.from_results:
        results = load_results_from_jsonl(args.from_results)
        if not results:
            print("Error: No results loaded from specified files", file=sys.stderr)
            sys.exit(1)
        n_files = len(args.from_results)
        print(f"Loaded {len(results)} results from {n_files} file pattern(s)", file=sys.stderr)
        display_from_results_stats(results)
        return

    # Determine input file
    if args.input is None:
        input_path = auto_detect_input()
        if input_path is None:
            print("Error: No input file specified and none detected", file=sys.stderr)
            sys.exit(1)
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {input_path}...", file=sys.stderr)

    # Load combined JSON (grouped by model)
    with open(input_path, encoding="utf-8") as f:
        grouped_data = json.load(f)

    # Set default reasoning effort for o-series models
    reasoning_effort = args.reasoning_effort
    if reasoning_effort is None and args.model.startswith("o"):
        reasoning_effort = "medium"

    # Initialize classifier
    classifier = LabelRecoveryClassifier(
        model=args.model,
        provider=args.provider,
        reasoning_effort=reasoning_effort,
    )

    # Flatten all conversations with their generator model
    all_convs: list[tuple[str, int, dict]] = []
    for generator_model, convs in grouped_data.items():
        for idx, conv in enumerate(convs):
            all_convs.append((generator_model, idx, conv))

    total = len(all_convs)
    print(f"Classifying {total} conversations with {args.model}...", file=sys.stderr)

    # Progress callback
    def on_progress(completed: int, total: int) -> None:
        print(f"\r  Progress: {completed}/{total}", end="", file=sys.stderr)

    # Classify all conversations
    results = classifier.classify_batch(
        all_convs,
        workers=args.workers,
        on_complete=on_progress,
        source_file=str(input_path),
    )
    print("", file=sys.stderr)  # Newline after progress

    # Print stats if requested
    if args.stats:
        print_stats(results)

        # Group by model and display table
        by_model: dict[str, list[ClassificationResult]] = {}
        for r in results:
            model = r.generator_model
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(r)
        classifier.display_results_table(by_model)

    # Determine output path
    if args.output is None:
        lang = extract_language(input_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = get_judge_model_filename(args.model, reasoning_effort)
        output_path = Path(f"data/label_recovery_{model_name}_{lang}_{timestamp}.jsonl")
    else:
        output_path = Path(args.output)

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} classification(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
