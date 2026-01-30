#!/usr/bin/env python3
"""
Show Model Metrics Statistics

Reads conversation metrics CSV files and displays paper metrics
(intra-model similarity, self-BLEU, TTR/MATTR) in a compact format.

Usage:
    python show_intra_similarity.py                           # Show intra-model similarity
    python show_intra_similarity.py --metric ttr              # Show TTR/MATTR
    python show_intra_similarity.py --metric self-bleu        # Show self-BLEU
    python show_intra_similarity.py -l hu --format table      # Detailed table
"""

import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def format_similarity(avg: float, std: float = None, precision: int = 2) -> str:
    """Format value as 'avg±std' with specified precision, or just 'avg' if std is None."""
    if std is not None:
        return f"{avg:.{precision}f}±{std:.{precision}f}"
    else:
        return f"{avg:.{precision}f}"


def read_metrics_file(language: str) -> pd.DataFrame:
    """Read the metrics CSV file for the specified language."""
    language_files = {
        'et': 'estonian_conversation_metrics.csv',
        'hu': 'hungarian_conversation_metrics.csv',
        'fi': 'finnish_conversation_metrics.csv'
    }
    
    if language not in language_files:
        raise ValueError(f"Unknown language: {language}. Use: et, hu, or fi")
    
    csv_file = Path(language_files[language])
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    return pd.read_csv(csv_file)


def get_similarity_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extract intra-model similarity statistics per model."""
    if 'model_name' not in df.columns:
        raise ValueError("CSV file missing 'model_name' column")
    
    required_cols = ['avg_similarity', 'std_similarity', 'min_similarity', 'max_similarity', 'median_similarity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"CSV file missing columns: {missing_cols}")
    
    # Group by model and get first row (all rows have same similarity values per model)
    stats = df.groupby('model_name').first()[required_cols].reset_index()
    
    # Sort by model name
    stats = stats.sort_values('model_name')
    
    return stats


def get_self_bleu_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extract self-BLEU statistics per model."""
    if 'model_name' not in df.columns:
        raise ValueError("CSV file missing 'model_name' column")
    
    # Try both column naming conventions (with and without dataset_ prefix and _self_bleu suffix)
    full_col = None
    agent_col = None
    client_col = None
    
    if 'full_self_bleu' in df.columns:
        full_col = 'full_self_bleu'
    elif 'dataset_full_conversation_self_bleu_self_bleu' in df.columns:
        full_col = 'dataset_full_conversation_self_bleu_self_bleu'
    
    if 'agent_self_bleu' in df.columns:
        agent_col = 'agent_self_bleu'
    elif 'dataset_agent_response_self_bleu_self_bleu' in df.columns:
        agent_col = 'dataset_agent_response_self_bleu_self_bleu'
    
    if 'client_self_bleu' in df.columns:
        client_col = 'client_self_bleu'
    elif 'dataset_client_response_self_bleu_self_bleu' in df.columns:
        client_col = 'dataset_client_response_self_bleu_self_bleu'
    
    if not full_col or not agent_col or not client_col:
        missing = []
        if not full_col:
            missing.append('full_self_bleu or dataset_full_conversation_self_bleu_self_bleu')
        if not agent_col:
            missing.append('agent_self_bleu or dataset_agent_response_self_bleu_self_bleu')
        if not client_col:
            missing.append('client_self_bleu or dataset_client_response_self_bleu_self_bleu')
        raise ValueError(f"CSV file missing columns: {missing}")
    
    # Group by model and get first row (all rows have same self-BLEU values per model)
    # Use consistent column names for output
    stats = df.groupby('model_name').first()[[full_col, agent_col, client_col]].reset_index()
    stats.columns = ['model_name', 'full_self_bleu', 'agent_self_bleu', 'client_self_bleu']
    
    # Sort by model name
    stats = stats.sort_values('model_name')
    
    return stats

def get_ttr_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TTR (simple and MATTR) statistics per model from per-conversation values."""
    if 'model_name' not in df.columns:
        raise ValueError("CSV file missing 'model_name' column")
    
    # Try all column naming conventions
    ttr_col = None
    mattr_col = None
    
    if 'ttr' in df.columns:
        ttr_col = 'ttr'
    elif 'type_token_ratio' in df.columns:
        ttr_col = 'type_token_ratio'
    elif 'lexical_diversity_type_token_ratio' in df.columns:
        ttr_col = 'lexical_diversity_type_token_ratio'
    
    if 'mattr' in df.columns:
        mattr_col = 'mattr'
    elif 'moving_avg_ttr' in df.columns:
        mattr_col = 'moving_avg_ttr'
    elif 'lexical_diversity_moving_avg_ttr' in df.columns:
        mattr_col = 'lexical_diversity_moving_avg_ttr'
    
    if not ttr_col or not mattr_col:
        missing = []
        if not ttr_col:
            missing.append('ttr, type_token_ratio, or lexical_diversity_type_token_ratio')
        if not mattr_col:
            missing.append('mattr, moving_avg_ttr, or lexical_diversity_moving_avg_ttr')
        raise ValueError(f"CSV file missing columns: {missing}")
    
    # Calculate mean, std, min, max per model for both TTR types
    # Use consistent column names regardless of input naming
    stats = df.groupby('model_name').agg({
        ttr_col: [('mean_ttr', 'mean'), ('std_ttr', 'std'), ('min_ttr', 'min'), ('max_ttr', 'max'), ('median_ttr', 'median')],
        mattr_col: [('mean_mattr', 'mean'), ('std_mattr', 'std'), ('min_mattr', 'min'), ('max_mattr', 'max'), ('median_mattr', 'median')]
    })
    
    # Flatten column names - always use simple column names for consistency
    stats.columns = ['_'.join(col).strip() if col[0] == 'model_name' else col[1] for col in stats.columns.values]
    stats = stats.reset_index()
    
    # Sort by model name
    stats = stats.sort_values('model_name')
    
    return stats


def display_compact(stats: pd.DataFrame, language: str, metric: str):
    """Display in compact format: model | avg±std or just avg"""
    language_names = {'et': '🇪🇪 Estonian', 'hu': '🇭🇺 Hungarian', 'fi': '🇫🇮 Finnish'}
    metric_names = {
        'similarity': 'Intra-Model Similarity',
        'self-bleu': 'Self-BLEU Scores',
        # 'coherence': 'Semantic Coherence',  # Not needed for paper
        'ttr': 'Type-Token Ratio (TTR and MATTR)'
    }
    
    console.print(f"\n[bold cyan]{language_names.get(language, language)} {metric_names[metric]}[/bold cyan]\n")
    
    for _, row in stats.iterrows():
        model_name = row['model_name']
        
        if metric == 'similarity':
            value = format_similarity(row['avg_similarity'], row['std_similarity'])
            console.print(f"│ {model_name:<40} │  {value}")
        
        elif metric == 'self-bleu':
            full = format_similarity(row['full_self_bleu'])
            agent = format_similarity(row['agent_self_bleu'])
            client = format_similarity(row['client_self_bleu'])
            console.print(f"│ {model_name:<40} │  Full: {full}  Agent: {agent}  Client: {client}")
        
        # elif metric == 'coherence':  # Not needed for paper
        #     value = format_similarity(row['mean_coherence'], row['std_coherence'])
        #     console.print(f"│ {model_name:<40} │  {value}")
        
        elif metric == 'ttr':
            ttr_val = format_similarity(row['mean_ttr'], row['std_ttr'])
            mattr_val = format_similarity(row['mean_mattr'], row['std_mattr'])
            console.print(f"│ {model_name:<40} │  TTR: {ttr_val}  MATTR: {mattr_val}")
    
    console.print()


def display_table(stats: pd.DataFrame, language: str, metric: str):
    """Display in detailed table format."""
    language_names = {'et': '🇪🇪 Estonian', 'hu': '🇭🇺 Hungarian', 'fi': '🇫🇮 Finnish'}
    metric_names = {
        'similarity': 'Intra-Model Similarity Statistics',
        'self-bleu': 'Self-BLEU Statistics',
        # 'coherence': 'Semantic Coherence Statistics',  # Not needed for paper
        'ttr': 'Type-Token Ratio Statistics'
    }
    
    table = Table(
        title=f"{language_names.get(language, language)} {metric_names[metric]}",
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Model", style="cyan", no_wrap=True)
    
    if metric == 'similarity':
        table.add_column("Avg ± Std", justify="center")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Median", justify="right")
        
        for _, row in stats.iterrows():
            table.add_row(
                row['model_name'],
                format_similarity(row['avg_similarity'], row['std_similarity']),
                f"{row['min_similarity']:.4f}",
                f"{row['max_similarity']:.4f}",
                f"{row['median_similarity']:.4f}"
            )
    
    elif metric == 'self-bleu':
        table.add_column("Full", justify="center")
        table.add_column("Agent", justify="center")
        table.add_column("Client", justify="center")
        
        for _, row in stats.iterrows():
            table.add_row(
                row['model_name'],
                f"{row['full_self_bleu']:.4f}",
                f"{row['agent_self_bleu']:.4f}",
                f"{row['client_self_bleu']:.4f}"
            )
    
    elif metric == 'ttr':
        table.add_column("TTR Mean ± Std", justify="center")
        table.add_column("MATTR Mean ± Std", justify="center")
        table.add_column("TTR Min/Max", justify="center")
        table.add_column("MATTR Min/Max", justify="center")
        
        for _, row in stats.iterrows():
            table.add_row(
                row['model_name'],
                format_similarity(row['mean_ttr'], row['std_ttr']),
                format_similarity(row['mean_mattr'], row['std_mattr']),
                f"{row['min_ttr']:.3f}/{row['max_ttr']:.3f}",
                f"{row['min_mattr']:.3f}/{row['max_mattr']:.3f}"
            )
    
    console.print()
    console.print(table)
    console.print()


def display_detailed(stats: pd.DataFrame, language: str, metric: str):
    """Display in detailed multi-line format."""
    language_names = {'et': '🇪🇪 Estonian', 'hu': '🇭🇺 Hungarian', 'fi': '🇫🇮 Finnish'}
    metric_names = {
        'similarity': 'Intra-Model Similarity',
        'self-bleu': 'Self-BLEU Scores',
        # 'coherence': 'Semantic Coherence',  # Not needed for paper
        'ttr': 'Type-Token Ratio (TTR and MATTR)'
    }
    
    console.print(f"\n[bold cyan]=== {language_names.get(language, language)} {metric_names[metric]} ===[/bold cyan]\n")
    
    for _, row in stats.iterrows():
        console.print(f"[cyan]{row['model_name']}:[/cyan]")
        
        if metric == 'similarity':
            console.print(f"  Avg Similarity: {row['avg_similarity']:.4f}")
            console.print(f"  Std Similarity: {row['std_similarity']:.4f}")
            console.print(f"  Min Similarity: {row['min_similarity']:.4f}")
            console.print(f"  Max Similarity: {row['max_similarity']:.4f}")
            console.print(f"  Median Similarity: {row['median_similarity']:.4f}")
        
        elif metric == 'self-bleu':
            console.print(f"  Full Self-BLEU: {row['full_self_bleu']:.4f}")
            console.print(f"  Agent Self-BLEU: {row['agent_self_bleu']:.4f}")
            console.print(f"  Client Self-BLEU: {row['client_self_bleu']:.4f}")

        elif metric == 'ttr':
            console.print(f"  Simple TTR (Mean): {row['mean_ttr']:.4f}")
            console.print(f"  Simple TTR (Std): {row['std_ttr']:.4f}")
            console.print(f"  Simple TTR (Min/Max): {row['min_ttr']:.4f} / {row['max_ttr']:.4f}")
            console.print(f"  MATTR (Mean): {row['mean_mattr']:.4f}")
            console.print(f"  MATTR (Std): {row['std_mattr']:.4f}")
            console.print(f"  MATTR (Min/Max): {row['min_mattr']:.4f} / {row['max_mattr']:.4f}")
        
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description='Display metrics statistics from conversation metrics CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python show_intra_similarity.py                          # Show intra-model similarity
    python show_intra_similarity.py --metric ttr             # Show TTR and MATTR
    python show_intra_similarity.py --metric self-bleu       # Show self-BLEU
    python show_intra_similarity.py -l hu --format table     # Detailed table
    python show_intra_similarity.py -l et --format detailed  # Detailed view
        """
    )
    
    parser.add_argument(
        '-l', '--language',
        type=str,
        choices=['et', 'hu', 'fi', 'all'],
        default='all',
        help='Language to display: et (Estonian), hu (Hungarian), fi (Finnish), or all (default: all)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        choices=['similarity', 'self-bleu', 'ttr'],
        default='similarity',
        help='Metric to display: similarity (default), self-bleu, or ttr'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['compact', 'table', 'detailed'],
        default='compact',
        help='Output format: compact (default), table, or detailed'
    )
    
    parser.add_argument(
        '--precision',
        type=int,
        default=2,
        help='Decimal precision for values (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Determine which languages to process
    languages = ['et', 'hu', 'fi'] if args.language == 'all' else [args.language]
    
    # Select stats extraction function based on metric
    stats_funcs = {
        'similarity': get_similarity_stats,
        'self-bleu': get_self_bleu_stats,
        # 'coherence': get_coherence_stats,  # Not needed for paper
        'ttr': get_ttr_stats
    }
    
    get_stats_func = stats_funcs[args.metric]
    
    # Display format functions
    display_funcs = {
        'compact': display_compact,
        'table': display_table,
        'detailed': display_detailed
    }
    
    display_func = display_funcs[args.format]
    
    # Process each language
    for lang in languages:
        try:
            df = read_metrics_file(lang)
            stats = get_stats_func(df)
            display_func(stats, lang, args.metric)
            
        except FileNotFoundError as e:
            console.print(f"[yellow]⚠️  Skipping {lang}: {e}[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ Error processing {lang}: {e}[/red]")
    
    # Display interpretation guide based on metric
    if args.format != 'detailed':
        if args.metric == 'similarity':
            console.print("[dim]Note: Lower similarity = more diverse conversations within model[/dim]")
            console.print("[dim]      Higher similarity = more formulaic/template-like conversations[/dim]")
        elif args.metric == 'self-bleu':
            console.print("[dim]Note: Lower Self-BLEU = more diverse (better)[/dim]")
            console.print("[dim]      Higher Self-BLEU = more formulaic/repetitive (worse)[/dim]")
        # elif args.metric == 'coherence':  # Not needed for paper
        #     console.print("[dim]Note: Higher coherence = better topical continuity[/dim]")
        #     console.print("[dim]      Lower coherence = more topic drift[/dim]")
        elif args.metric == 'ttr':
            console.print("[dim]Note: TTR = simple Type-Token Ratio (length-dependent)[/dim]")
            console.print("[dim]      MATTR = Moving Average TTR over 100-token windows (length-independent)[/dim]")
            console.print("[dim]      Higher values = more lexical diversity[/dim]")


if __name__ == "__main__":
    main()
