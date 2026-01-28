"""Data models for ranking inversion analysis."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RankingResult:
    """Per-model ranking result for a specific language and metric."""
    language: str
    model: str
    metric: str
    mean_score: float
    rank: int
    n_conversations: int


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval results."""
    mean: float
    ci_low: float
    ci_high: float
    std: float


@dataclass
class PairwiseStats:
    """Statistics for comparing rankings between two languages."""
    metric: str
    language_pair: str
    kendall_tau: float
    kendall_p: float
    spearman_rho: float
    spearman_p: float
    inversions: int
    total_pairs: int
    inversion_rate: float

    # Bootstrap results
    tau_bootstrap: Optional[BootstrapResult] = None
    rho_bootstrap: Optional[BootstrapResult] = None

    # Permutation test results
    inversion_perm_p: Optional[float] = None


@dataclass
class InversionAnalysis:
    """Complete ranking inversion analysis results."""
    pairwise_stats: list[PairwiseStats]
    ranking_results: list[RankingResult]
    metadata: dict

    @property
    def languages(self) -> list[str]:
        """Get unique languages from the analysis."""
        return sorted(set(result.language for result in self.ranking_results))

    @property
    def metrics(self) -> list[str]:
        """Get unique metrics from the analysis."""
        return sorted(set(result.metric for result in self.ranking_results))

    @property
    def models(self) -> list[str]:
        """Get unique models from the analysis."""
        return sorted(set(result.model for result in self.ranking_results))

    def get_pairwise_stats(self, metric: str, language_pair: str) -> Optional[PairwiseStats]:
        """Get pairwise statistics for a specific metric and language pair."""
        for stats in self.pairwise_stats:
            if stats.metric == metric and stats.language_pair == language_pair:
                return stats
        return None

    def get_ranking_results(self, language: str, metric: str) -> list[RankingResult]:
        """Get ranking results for a specific language and metric."""
        return [
            result for result in self.ranking_results
            if result.language == language and result.metric == metric
        ]