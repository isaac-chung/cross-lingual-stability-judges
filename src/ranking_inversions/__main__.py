"""CLI entry point for ranking inversions analysis."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table

from .analyzer import RankingInversionAnalyzer


def display_results(analysis_data: Dict[str, Any], console: Console):
    """Display analysis results using Rich tables."""
    console.print("\n[bold blue]Ranking Inversion Analysis Results[/bold blue]\n")

    # Display metadata
    metadata = analysis_data.get('metadata', {})
    console.print(f"Languages: {', '.join(metadata.get('languages', []))}")
    console.print(f"Metrics: {', '.join(metadata.get('metrics', []))}")
    console.print(f"Total models: {metadata.get('total_models', 0)}")
    console.print(f"Total conversations: {metadata.get('total_conversations', 0)}")

    if metadata.get('n_bootstrap', 0) > 0:
        console.print(f"Bootstrap iterations: {metadata['n_bootstrap']}")
    if metadata.get('n_permutations', 0) > 0:
        console.print(f"Permutation iterations: {metadata['n_permutations']}")

    console.print()

    # Group pairwise stats by metric
    pairwise_stats = analysis_data.get('pairwise_stats', [])
    metrics = sorted(set(stat['metric'] for stat in pairwise_stats))

    for metric in metrics:
        metric_stats = [stat for stat in pairwise_stats if stat['metric'] == metric]
        if not metric_stats:
            continue

        console.print(f"[bold green]{metric} - Pairwise Ranking Comparisons[/bold green]")

        # Create table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Language Pair", style="cyan")
        table.add_column("Kendall τ", justify="right")
        table.add_column("τ p-value", justify="right")
        table.add_column("Spearman ρ", justify="right")
        table.add_column("ρ p-value", justify="right")
        table.add_column("Inversions", justify="right")
        table.add_column("Inversion Rate", justify="right")

        # Add bootstrap CI columns if available
        has_bootstrap = any('tau_bootstrap' in stat for stat in metric_stats)
        if has_bootstrap:
            table.add_column("τ CI", justify="right")
            table.add_column("ρ CI", justify="right")

        # Add permutation test column if available
        has_permutation = any('inversion_perm_p' in stat for stat in metric_stats)
        if has_permutation:
            table.add_column("Inv. p-value", justify="right")

        # Add rows
        for stat in sorted(metric_stats, key=lambda x: x['language_pair']):
            row = [
                stat['language_pair'],
                f"{stat['kendall_tau']:.3f}",
                f"{stat['kendall_p']:.4f}" if stat['kendall_p'] < 0.001 else f"{stat['kendall_p']:.3f}",
                f"{stat['spearman_rho']:.3f}",
                f"{stat['spearman_p']:.4f}" if stat['spearman_p'] < 0.001 else f"{stat['spearman_p']:.3f}",
                f"{stat['inversions']}/{stat['total_pairs']}",
                f"{stat['inversion_rate']:.3f}"
            ]

            if has_bootstrap:
                if 'tau_bootstrap' in stat and stat['tau_bootstrap']:
                    tau_ci = f"[{stat['tau_bootstrap']['ci_low']:.2f}, {stat['tau_bootstrap']['ci_high']:.2f}]"
                else:
                    tau_ci = "—"

                if 'rho_bootstrap' in stat and stat['rho_bootstrap']:
                    rho_ci = f"[{stat['rho_bootstrap']['ci_low']:.2f}, {stat['rho_bootstrap']['ci_high']:.2f}]"
                else:
                    rho_ci = "—"

                row.extend([tau_ci, rho_ci])

            if has_permutation:
                if 'inversion_perm_p' in stat:
                    perm_p = stat['inversion_perm_p']
                    if perm_p < 0.001:
                        row.append(f"{perm_p:.4f}")
                    else:
                        row.append(f"{perm_p:.3f}")
                else:
                    row.append("—")

            table.add_row(*row)

        console.print(table)
        console.print()

    # Display ranking results summary
    ranking_results = analysis_data.get('ranking_results', [])
    if ranking_results:
        console.print("[bold green]Model Rankings by Language and Metric[/bold green]")

        # Group by metric and language
        languages = sorted(set(result['language'] for result in ranking_results))
        for metric in metrics:
            metric_results = [result for result in ranking_results if result['metric'] == metric]
            if not metric_results:
                continue

            console.print(f"\n[bold]{metric}[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Model", style="cyan")

            for lang in languages:
                table.add_column(f"{lang.upper()}", justify="center")

            # Get all models for this metric
            models = sorted(set(result['model'] for result in metric_results))

            for model in models:
                row = [model]
                for lang in languages:
                    # Find result for this model and language
                    result = next(
                        (r for r in metric_results if r['model'] == model and r['language'] == lang),
                        None
                    )
                    if result:
                        rank_str = f"#{result['rank']} ({result['mean_score']:.3f})"
                        row.append(rank_str)
                    else:
                        row.append("—")

                table.add_row(*row)

            console.print(table)


def load_analysis_results(path: Path) -> Dict[str, Any]:
    """Load previously saved analysis results."""
    with open(path, 'r') as f:
        return json.load(f)


def auto_detect_languages(paths: List[Path]) -> List[str]:
    """Auto-detect languages from file paths."""
    languages = set()
    for path in paths:
        # Try to extract language code from filename
        import re
        match = re.search(r'[_-]([a-z]{2})[_.]', str(path))
        if match:
            languages.add(match.group(1))

        # Also try to peek into files for source_file patterns
        try:
            with open(path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    if 'source_file' in data:
                        source_match = re.search(r'combined_([a-z]{2})\.json', data['source_file'])
                        if source_match:
                            languages.add(source_match.group(1))
        except:
            continue

    return sorted(languages)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze ranking inversions across languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze judge results
  python -m ranking_inversions data/judge_*.jsonl --languages et fi hu en

  # Display saved results
  python -m ranking_inversions --from-results analysis.json

  # Combined analysis with label recovery
  python -m ranking_inversions --judge data/judge_*.jsonl \\
                                --label-recovery data/label_recovery_*.jsonl \\
                                --ground-truth combined.json \\
                                --languages et fi hu en

  # Quick analysis without bootstrap/permutation
  python -m ranking_inversions data/judge_*.jsonl --no-bootstrap --no-permutation
        """
    )

    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'judge_files',
        nargs='*',
        help='JSONL files containing llm_judge results'
    )
    input_group.add_argument(
        '--from-results',
        type=Path,
        help='Load and display previously saved analysis results'
    )

    # Combined mode
    parser.add_argument('--judge', nargs='+', help='Judge JSONL files (for combined mode)')
    parser.add_argument('--label-recovery', nargs='+', help='Label recovery JSONL files')
    parser.add_argument('--ground-truth', type=Path, help='Ground truth JSON file')

    # Analysis options
    parser.add_argument(
        '--languages',
        nargs='+',
        help='Languages to compare (default: auto-detect from files)'
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['G', 'R', 'C', 'F'],
        help='Metrics to analyze (default: G R C F)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=2000,
        help='Number of bootstrap iterations (default: 2000)'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=5000,
        help='Number of permutation iterations (default: 5000)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--no-bootstrap',
        action='store_true',
        help='Skip bootstrap confidence intervals (faster)'
    )
    parser.add_argument(
        '--no-permutation',
        action='store_true',
        help='Skip permutation tests (faster)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    console = Console()

    # Handle display mode
    if args.from_results:
        if not args.from_results.exists():
            console.print(f"[red]Error: Results file not found: {args.from_results}[/red]")
            sys.exit(1)

        try:
            results = load_analysis_results(args.from_results)
            display_results(results, console)
        except Exception as e:
            console.print(f"[red]Error loading results: {e}[/red]")
            sys.exit(1)
        return

    # Handle analysis mode
    analyzer = RankingInversionAnalyzer(
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed
    )

    try:
        # Determine input files
        judge_files = []
        if args.judge:
            judge_files.extend(Path(p) for p in args.judge)
        elif args.judge_files:
            judge_files.extend(Path(p) for p in args.judge_files)

        if not judge_files:
            console.print("[red]Error: No judge files specified[/red]")
            sys.exit(1)

        # Load data
        console.print("Loading judge results...")
        df_parts = []

        judge_df = analyzer.load_judge_results(judge_files)
        df_parts.append(judge_df)

        # Add label recovery if specified
        if args.label_recovery and args.ground_truth:
            console.print("Loading label recovery results...")
            lr_files = [Path(p) for p in args.label_recovery]
            lr_df = analyzer.load_label_recovery_results(lr_files, args.ground_truth)
            df_parts.append(lr_df)

            # Add LRA to metrics if not present
            if 'LRA' not in args.metrics:
                args.metrics = args.metrics + ['LRA']

        # Combine dataframes
        df = pd.concat(df_parts, ignore_index=True)

        # Auto-detect languages if not specified
        if not args.languages:
            args.languages = auto_detect_languages(judge_files)
            console.print(f"Auto-detected languages: {', '.join(args.languages)}")

        if not args.languages:
            console.print("[red]Error: No languages detected. Please specify --languages[/red]")
            sys.exit(1)

        # Filter metrics based on available data
        available_metrics = sorted(df['metric'].unique())
        requested_metrics = [m for m in args.metrics if m in available_metrics]

        if not requested_metrics:
            console.print(f"[red]Error: None of the requested metrics {args.metrics} found in data.[/red]")
            console.print(f"Available metrics: {available_metrics}")
            sys.exit(1)

        console.print(f"Analyzing metrics: {', '.join(requested_metrics)}")
        console.print(f"Languages: {', '.join(args.languages)}")

        # Run analysis
        console.print("Running analysis...")
        analysis = analyzer.analyze(
            df,
            metrics=requested_metrics,
            languages=args.languages,
            run_bootstrap=not args.no_bootstrap,
            run_permutation=not args.no_permutation
        )

        # Save results if requested
        if args.output:
            console.print(f"Saving results to {args.output}")
            analyzer.save_results(analysis, args.output)

        # Convert to dict for display
        analysis_dict = {
            'pairwise_stats': [],
            'ranking_results': [],
            'metadata': analysis.metadata
        }

        for stats in analysis.pairwise_stats:
            stats_dict = {
                'metric': stats.metric,
                'language_pair': stats.language_pair,
                'kendall_tau': stats.kendall_tau,
                'kendall_p': stats.kendall_p,
                'spearman_rho': stats.spearman_rho,
                'spearman_p': stats.spearman_p,
                'inversions': stats.inversions,
                'total_pairs': stats.total_pairs,
                'inversion_rate': stats.inversion_rate,
            }

            if stats.tau_bootstrap:
                stats_dict['tau_bootstrap'] = {
                    'mean': stats.tau_bootstrap.mean,
                    'ci_low': stats.tau_bootstrap.ci_low,
                    'ci_high': stats.tau_bootstrap.ci_high,
                    'std': stats.tau_bootstrap.std
                }

            if stats.rho_bootstrap:
                stats_dict['rho_bootstrap'] = {
                    'mean': stats.rho_bootstrap.mean,
                    'ci_low': stats.rho_bootstrap.ci_low,
                    'ci_high': stats.rho_bootstrap.ci_high,
                    'std': stats.rho_bootstrap.std
                }

            if stats.inversion_perm_p is not None:
                stats_dict['inversion_perm_p'] = stats.inversion_perm_p

            analysis_dict['pairwise_stats'].append(stats_dict)

        for result in analysis.ranking_results:
            analysis_dict['ranking_results'].append({
                'language': result.language,
                'model': result.model,
                'metric': result.metric,
                'mean_score': result.mean_score,
                'rank': result.rank,
                'n_conversations': result.n_conversations
            })

        # Display results
        display_results(analysis_dict, console)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()