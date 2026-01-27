"""CLI entry point for LLM-as-a-Judge evaluation."""

import argparse
import glob
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .evaluator import LLMJudge


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate conversations using LLM-as-a-Judge",
        prog="python -m llm_judge",
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
        help="Output JSONL path (default: data/judge_{lang}_{datetime}.jsonl)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Judge model (default: gpt-5-mini)",
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
        help="Reasoning effort for o-series models (default: medium for o-series)",
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
        help="Print summary statistics after evaluation",
    )
    parser.add_argument(
        "--from-results",
        nargs="+",
        metavar="FILE",
        help="Load existing JSONL results and display analysis (no evaluation)",
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
    # combined_et.json -> et
    stem = input_path.stem
    if stem.startswith("combined_"):
        return stem[9:]  # Remove "combined_" prefix
    return "unknown"


def print_stats(results: list[dict], generator_models: list[str]) -> None:
    """Print summary statistics."""
    print("\n--- Evaluation Statistics ---", file=sys.stderr)
    print(f"Total evaluated: {len(results)}", file=sys.stderr)

    # Group by generator model
    by_model: dict[str, list[dict]] = {}
    for r in results:
        model = r["generator_model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    print("\nScores by generator model:", file=sys.stderr)
    for model in sorted(by_model.keys()):
        model_results = by_model[model]
        n = len(model_results)

        avg_g = sum(r["G"] for r in model_results) / n
        avg_r = sum(r["R"] for r in model_results) / n
        avg_c = sum(r["C"] for r in model_results) / n
        avg_f = sum(r["F"] for r in model_results) / n

        print(f"\n  {model} (n={n}):", file=sys.stderr)
        print(f"    G: {avg_g:.2f}  R: {avg_r:.2f}  C: {avg_c:.2f}  F: {avg_f:.2f}", file=sys.stderr)


def load_results_from_jsonl(paths: list[str]) -> list[dict]:
    """Load evaluation results from JSONL files.

    Args:
        paths: List of file paths (supports glob patterns).

    Returns:
        List of result dicts.
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
                    results.append(json.loads(line))
    return results


def analyze_results(results: list[dict]) -> dict[str, dict]:
    """Compute detailed statistics per generator model.

    Args:
        results: List of evaluation result dicts.

    Returns:
        Dict mapping model name to statistics dict.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        model = r.get("generator_model", "unknown")
        by_model[model].append(r)

    stats = {}
    for model, model_results in by_model.items():
        scores = {
            "G": np.array([r["G"] for r in model_results]),
            "R": np.array([r["R"] for r in model_results]),
            "C": np.array([r["C"] for r in model_results]),
            "F": np.array([r["F"] for r in model_results]),
        }
        model_stats = {"n": len(model_results)}
        for metric, values in scores.items():
            model_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": int(np.min(values)),
                "max": int(np.max(values)),
                "distribution": {int(i): int(np.sum(values == i)) for i in range(int(np.max(values)) + 1)},
            }
        stats[model] = model_stats

    return stats


def display_model_comparison_table(stats: dict[str, dict]) -> None:
    """Display model comparison table with mean scores.

    Args:
        stats: Dict from analyze_results().
    """
    print("\n" + "=" * 70, file=sys.stderr)
    print("MODEL COMPARISON (Mean Scores)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # Header
    print(f"{'Model':<30} {'G':>8} {'R':>8} {'C':>8} {'F':>8} {'Total':>8}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    for model in sorted(stats.keys()):
        m = stats[model]
        g = m["G"]["mean"]
        r = m["R"]["mean"]
        c = m["C"]["mean"]
        f = m["F"]["mean"]
        total = g + r + c + f
        print(f"{model:<30} {g:>8.2f} {r:>8.2f} {c:>8.2f} {f:>8.2f} {total:>8.2f}", file=sys.stderr)

    print("=" * 70, file=sys.stderr)


def print_detailed_stats(stats: dict[str, dict]) -> None:
    """Print detailed analysis of results.

    Args:
        stats: Dict from analyze_results().
    """
    print("\n" + "=" * 70, file=sys.stderr)
    print("DETAILED EVALUATION ANALYSIS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    for model in sorted(stats.keys()):
        model_stats = stats[model]
        n = model_stats["n"]

        print(f"\n{model} (n={n})", file=sys.stderr)
        print("-" * 50, file=sys.stderr)

        # Header
        print(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}  Distribution", file=sys.stderr)
        print("-" * 70, file=sys.stderr)

        for metric in ["G", "R", "C", "F"]:
            m = model_stats[metric]
            dist = m["distribution"]
            dist_str = " ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
            print(
                f"{metric:<12} {m['mean']:>8.2f} {m['std']:>8.2f} {m['min']:>6} {m['max']:>6}  {dist_str}",
                file=sys.stderr,
            )

    display_model_comparison_table(stats)


def main() -> None:
    """Main entry point."""
    load_dotenv()
    args = parse_args()

    # Handle --from-results mode (analysis only, no evaluation)
    if args.from_results:
        results = load_results_from_jsonl(args.from_results)
        if not results:
            print("Error: No results loaded from specified files", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(results)} results from {len(args.from_results)} file(s)", file=sys.stderr)
        stats = analyze_results(results)
        print_detailed_stats(stats)
        return

    # Determine input file
    if args.input is None:
        input_path = auto_detect_input()
        if input_path is None:
            print("Error: No input file specified and none auto-detected", file=sys.stderr)
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

    # Initialize judge
    judge = LLMJudge(
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
    print(f"Evaluating {total} conversations with {args.model}...", file=sys.stderr)

    # Progress callback
    def on_progress(completed: int, total: int) -> None:
        print(f"\r  Progress: {completed}/{total}", end="", file=sys.stderr)

    # Evaluate all conversations
    convs_only = [c[2] for c in all_convs]
    evaluations = judge.evaluate_batch(
        convs_only,
        workers=args.workers,
        on_complete=on_progress,
    )
    print("", file=sys.stderr)  # Newline after progress

    # Build results with metadata
    results = []
    for (generator_model, conv_idx, conv), evaluation in zip(all_convs, evaluations):
        results.append({
            "conversation_id": conv_idx,
            "generator_model": generator_model,
            "G": evaluation.G,
            "R": evaluation.R,
            "C": evaluation.C,
            "F": evaluation.F,
            "explanation": evaluation.explanation,
            "source_file": str(input_path),
        })

    # Print stats if requested
    if args.stats:
        print_stats(results, list(grouped_data.keys()))

    # Determine output path
    if args.output is None:
        lang = extract_language(input_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"data/judge_{lang}_{timestamp}.jsonl")
    else:
        output_path = Path(args.output)

    # Write results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} evaluation(s) to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
