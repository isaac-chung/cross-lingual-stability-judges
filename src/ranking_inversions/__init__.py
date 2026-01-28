"""Ranking inversions analysis module.

This module analyzes cross-language ranking stability using results from
llm_judge and label_recovery modules. It computes ranking correlations
(Kendall tau, Spearman rho) and pairwise inversions across language pairs.

Example usage:

    from ranking_inversions import RankingInversionAnalyzer

    analyzer = RankingInversionAnalyzer()
    judge_df = analyzer.load_judge_results(['data/judge_et.jsonl', 'data/judge_fi.jsonl'])
    analysis = analyzer.analyze(judge_df, metrics=['G', 'R', 'C', 'F'], languages=['et', 'fi'])

    for stats in analysis.pairwise_stats:
        print(f"{stats.language_pair}: τ={stats.kendall_tau:.3f}, inversions={stats.inversions}")
"""

from .analyzer import RankingInversionAnalyzer
from .models import (
    BootstrapResult,
    InversionAnalysis,
    PairwiseStats,
    RankingResult,
)

__all__ = [
    'RankingInversionAnalyzer',
    'InversionAnalysis',
    'PairwiseStats',
    'RankingResult',
    'BootstrapResult',
]

__version__ = '1.0.0'