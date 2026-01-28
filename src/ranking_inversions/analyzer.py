"""Core ranking inversion analysis functionality."""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy import stats

from .models import BootstrapResult, InversionAnalysis, PairwiseStats, RankingResult


class RankingInversionAnalyzer:
    """Analyzer for cross-language ranking stability using ranking inversions."""

    def __init__(
        self,
        n_bootstrap: int = 2000,
        n_permutations: int = 5000,
        seed: int = 42,
    ):
        """Initialize the analyzer.

        Args:
            n_bootstrap: Number of bootstrap iterations for confidence intervals
            n_permutations: Number of permutation iterations for significance tests
            seed: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(seed)

    def load_judge_results(self, paths: list[Union[str, Path]]) -> pd.DataFrame:
        """Load llm_judge JSONL files, extracting G, R, C, F scores per conversation.

        Args:
            paths: List of paths to JSONL files containing judge results

        Returns:
            DataFrame with columns: language, model, metric, conversation_id, score
        """
        records = []

        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Judge results file not found: {path}")

            with open(path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())

                    # Extract language from source_file path
                    source_file = data.get('source_file', str(path))
                    language_match = re.search(r'combined_([a-z]{2})\.json', source_file)
                    if language_match:
                        language = language_match.group(1)
                    else:
                        # Fallback: extract from filename
                        lang_match = re.search(r'[_-]([a-z]{2})[_.]', str(path))
                        language = lang_match.group(1) if lang_match else 'unknown'

                    model = data['generator_model']
                    conversation_id = data['conversation_id']

                    # Extract GRCF scores
                    for metric in ['G', 'R', 'C', 'F']:
                        if metric in data:
                            records.append({
                                'language': language,
                                'model': model,
                                'metric': metric,
                                'conversation_id': conversation_id,
                                'score': float(data[metric])
                            })

        if not records:
            raise ValueError("No valid judge results found in provided files")

        return pd.DataFrame(records)

    def load_label_recovery_results(
        self,
        paths: list[Union[str, Path]],
        ground_truth_path: Union[str, Path]
    ) -> pd.DataFrame:
        """Load label_recovery JSONL files, compute LRA per conversation.

        Args:
            paths: List of paths to JSONL files containing label recovery results
            ground_truth_path: Path to JSON file with ground truth labels

        Returns:
            DataFrame with columns: language, model, metric, conversation_id, score
        """
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)

        # Create lookup for ground truth
        gt_lookup = {}
        for item in ground_truth:
            conv_id = item['conversation_id']
            gt_lookup[conv_id] = {
                'industry': item['industry'],
                'problem': item['problem']
            }

        records = []

        for path in paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Label recovery results file not found: {path}")

            with open(path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())

                    # Extract language from source_file or filename
                    source_file = data.get('source_file', str(path))
                    language_match = re.search(r'combined_([a-z]{2})\.json', source_file)
                    if language_match:
                        language = language_match.group(1)
                    else:
                        lang_match = re.search(r'[_-]([a-z]{2})[_.]', str(path))
                        language = lang_match.group(1) if lang_match else 'unknown'

                    model = data['generator_model']
                    conversation_id = data['conversation_id']

                    # Check if predictions match ground truth
                    if conversation_id in gt_lookup:
                        gt = gt_lookup[conversation_id]
                        pred_industry = data.get('predicted_industry')
                        pred_problem = data.get('predicted_problem')

                        # Binary accuracy: both industry and problem must be correct
                        industry_correct = pred_industry == gt['industry']
                        problem_correct = pred_problem == gt['problem']
                        lra_score = 1.0 if (industry_correct and problem_correct) else 0.0

                        records.append({
                            'language': language,
                            'model': model,
                            'metric': 'LRA',
                            'conversation_id': conversation_id,
                            'score': lra_score
                        })

        if not records:
            raise ValueError("No valid label recovery results found in provided files")

        return pd.DataFrame(records)

    def compute_means_and_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-language/per-model means and assign ranks.

        Args:
            df: DataFrame with columns: language, model, metric, conversation_id, score

        Returns:
            DataFrame with ranking results
        """
        # Group by language, model, metric and compute means
        grouped = df.groupby(['language', 'model', 'metric']).agg({
            'score': ['mean', 'count']
        }).reset_index()

        # Flatten column names
        grouped.columns = ['language', 'model', 'metric', 'mean_score', 'n_conversations']

        # Assign ranks within each language-metric combination (higher scores get better ranks)
        ranking_results = []
        for (language, metric), group in grouped.groupby(['language', 'metric']):
            # Sort by mean_score descending, assign ranks
            sorted_group = group.sort_values('mean_score', ascending=False)
            for rank, (_, row) in enumerate(sorted_group.iterrows(), 1):
                ranking_results.append(RankingResult(
                    language=row['language'],
                    model=row['model'],
                    metric=row['metric'],
                    mean_score=row['mean_score'],
                    rank=rank,
                    n_conversations=row['n_conversations']
                ))

        return pd.DataFrame([r.__dict__ for r in ranking_results])

    def pairwise_rank_stats(
        self,
        means_df: pd.DataFrame,
        metric: str,
        languages: list[str]
    ) -> list[PairwiseStats]:
        """Compute Kendall τ, Spearman ρ, inversions for all language pairs.

        Args:
            means_df: DataFrame with ranking results
            metric: Metric to analyze
            languages: List of languages to compare

        Returns:
            List of pairwise statistics
        """
        results = []

        for lang1, lang2 in combinations(languages, 2):
            # Get rankings for each language
            ranks1 = means_df[
                (means_df['language'] == lang1) &
                (means_df['metric'] == metric)
            ].set_index('model')['rank'].to_dict()

            ranks2 = means_df[
                (means_df['language'] == lang2) &
                (means_df['metric'] == metric)
            ].set_index('model')['rank'].to_dict()

            # Find common models
            common_models = set(ranks1.keys()) & set(ranks2.keys())
            if len(common_models) < 2:
                continue

            # Extract rank vectors
            models = sorted(common_models)
            rank_vec1 = [ranks1[model] for model in models]
            rank_vec2 = [ranks2[model] for model in models]

            # Compute correlations
            tau, tau_p = stats.kendalltau(rank_vec1, rank_vec2)
            rho, rho_p = stats.spearmanr(rank_vec1, rank_vec2)

            # Count inversions
            inversions = 0
            total_pairs = 0
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    total_pairs += 1
                    # Check if relative ordering is different
                    order1 = rank_vec1[i] < rank_vec1[j]  # model i ranks better than model j in lang1
                    order2 = rank_vec2[i] < rank_vec2[j]  # model i ranks better than model j in lang2
                    if order1 != order2:
                        inversions += 1

            inversion_rate = inversions / total_pairs if total_pairs > 0 else 0.0

            results.append(PairwiseStats(
                metric=metric,
                language_pair=f"{lang1}-{lang2}",
                kendall_tau=tau,
                kendall_p=tau_p,
                spearman_rho=rho,
                spearman_p=rho_p,
                inversions=inversions,
                total_pairs=total_pairs,
                inversion_rate=inversion_rate
            ))

        return results

    def bootstrap_rank_corrs(
        self,
        df: pd.DataFrame,
        metric: str,
        languages: list[str]
    ) -> dict[str, dict[str, BootstrapResult]]:
        """Bootstrap confidence intervals for rank correlations.

        Args:
            df: Original score DataFrame
            metric: Metric to analyze
            languages: List of languages to compare

        Returns:
            Nested dict: {language_pair: {tau/rho: BootstrapResult}}
        """
        results = {}

        for lang1, lang2 in combinations(languages, 2):
            pair_key = f"{lang1}-{lang2}"

            # Get data for this metric and language pair
            df1 = df[(df['language'] == lang1) & (df['metric'] == metric)]
            df2 = df[(df['language'] == lang2) & (df['metric'] == metric)]

            # Find common models and conversations
            common_models = set(df1['model']) & set(df2['model'])
            if len(common_models) < 2:
                continue

            # Bootstrap iterations
            tau_samples = []
            rho_samples = []

            for _ in range(self.n_bootstrap):
                # For each model, resample conversations with replacement
                boot_df1_parts = []
                boot_df2_parts = []

                for model in common_models:
                    model_df1 = df1[df1['model'] == model]
                    model_df2 = df2[df2['model'] == model]

                    if len(model_df1) > 0 and len(model_df2) > 0:
                        # Sample conversations with replacement
                        boot_indices = self.rng.choice(len(model_df1), size=len(model_df1), replace=True)
                        boot_df1_parts.append(model_df1.iloc[boot_indices])

                        boot_indices = self.rng.choice(len(model_df2), size=len(model_df2), replace=True)
                        boot_df2_parts.append(model_df2.iloc[boot_indices])

                if not boot_df1_parts or not boot_df2_parts:
                    continue

                boot_df1 = pd.concat(boot_df1_parts, ignore_index=True)
                boot_df2 = pd.concat(boot_df2_parts, ignore_index=True)

                # Compute means and ranks for bootstrap sample
                boot_means1 = boot_df1.groupby('model')['score'].mean()
                boot_means2 = boot_df2.groupby('model')['score'].mean()

                boot_ranks1 = boot_means1.rank(ascending=False, method='average')
                boot_ranks2 = boot_means2.rank(ascending=False, method='average')

                # Only include common models
                common_in_boot = set(boot_ranks1.index) & set(boot_ranks2.index)
                if len(common_in_boot) < 2:
                    continue

                models = sorted(common_in_boot)
                rank_vec1 = [boot_ranks1[model] for model in models]
                rank_vec2 = [boot_ranks2[model] for model in models]

                # Compute correlations
                try:
                    tau, _ = stats.kendalltau(rank_vec1, rank_vec2)
                    rho, _ = stats.spearmanr(rank_vec1, rank_vec2)
                    if not np.isnan(tau):
                        tau_samples.append(tau)
                    if not np.isnan(rho):
                        rho_samples.append(rho)
                except:
                    continue

            # Compute bootstrap statistics
            if tau_samples:
                tau_mean = np.mean(tau_samples)
                tau_ci = np.percentile(tau_samples, [2.5, 97.5])
                tau_std = np.std(tau_samples)
                tau_result = BootstrapResult(tau_mean, tau_ci[0], tau_ci[1], tau_std)
            else:
                tau_result = None

            if rho_samples:
                rho_mean = np.mean(rho_samples)
                rho_ci = np.percentile(rho_samples, [2.5, 97.5])
                rho_std = np.std(rho_samples)
                rho_result = BootstrapResult(rho_mean, rho_ci[0], rho_ci[1], rho_std)
            else:
                rho_result = None

            results[pair_key] = {
                'tau': tau_result,
                'rho': rho_result
            }

        return results

    def permutation_test_inversions(
        self,
        means_df: pd.DataFrame,
        metric: str,
        languages: list[str]
    ) -> dict[str, float]:
        """Permutation test for statistical significance of inversions.

        Args:
            means_df: DataFrame with ranking results
            metric: Metric to analyze
            languages: List of languages to compare

        Returns:
            Dict mapping language pairs to p-values
        """
        results = {}

        # Get all data for this metric
        metric_data = means_df[means_df['metric'] == metric].copy()

        for lang1, lang2 in combinations(languages, 2):
            pair_key = f"{lang1}-{lang2}"

            # Get observed inversion count
            observed_stats = self.pairwise_rank_stats(metric_data, metric, [lang1, lang2])
            if not observed_stats:
                continue
            observed_inversions = observed_stats[0].inversions

            # Permutation test
            perm_inversions = []

            # Extract data for the two languages
            lang_data = metric_data[metric_data['language'].isin([lang1, lang2])].copy()

            for _ in range(self.n_permutations):
                # Shuffle language labels
                shuffled_data = lang_data.copy()
                shuffled_labels = self.rng.permutation(shuffled_data['language'].values)
                shuffled_data['language'] = shuffled_labels

                # Compute inversions for shuffled data
                perm_stats = self.pairwise_rank_stats(shuffled_data, metric, [lang1, lang2])
                if perm_stats:
                    perm_inversions.append(perm_stats[0].inversions)

            # Compute p-value (two-tailed)
            if perm_inversions:
                p_value = np.mean(np.array(perm_inversions) >= observed_inversions)
                results[pair_key] = min(2 * p_value, 1.0)  # Two-tailed

        return results

    def analyze(
        self,
        df: pd.DataFrame,
        metrics: list[str],
        languages: list[str],
        run_bootstrap: bool = True,
        run_permutation: bool = True
    ) -> InversionAnalysis:
        """Run complete analysis pipeline.

        Args:
            df: Score DataFrame
            metrics: List of metrics to analyze
            languages: List of languages to compare
            run_bootstrap: Whether to compute bootstrap confidence intervals
            run_permutation: Whether to run permutation tests

        Returns:
            Complete analysis results
        """
        # Compute means and ranks
        means_df = self.compute_means_and_ranks(df)

        # Convert back to RankingResult objects
        ranking_results = [
            RankingResult(**row) for _, row in means_df.iterrows()
        ]

        # Compute pairwise statistics for all metrics
        all_pairwise_stats = []

        for metric in metrics:
            # Basic pairwise statistics
            pairwise_stats = self.pairwise_rank_stats(means_df, metric, languages)

            # Bootstrap confidence intervals
            if run_bootstrap:
                bootstrap_results = self.bootstrap_rank_corrs(df, metric, languages)
                for stats in pairwise_stats:
                    pair_key = stats.language_pair
                    if pair_key in bootstrap_results:
                        stats.tau_bootstrap = bootstrap_results[pair_key]['tau']
                        stats.rho_bootstrap = bootstrap_results[pair_key]['rho']

            # Permutation tests
            if run_permutation:
                perm_results = self.permutation_test_inversions(means_df, metric, languages)
                for stats in pairwise_stats:
                    pair_key = stats.language_pair
                    if pair_key in perm_results:
                        stats.inversion_perm_p = perm_results[pair_key]

            all_pairwise_stats.extend(pairwise_stats)

        # Compile metadata
        metadata = {
            'languages': languages,
            'metrics': metrics,
            'n_bootstrap': self.n_bootstrap if run_bootstrap else 0,
            'n_permutations': self.n_permutations if run_permutation else 0,
            'total_conversations': len(df['conversation_id'].unique()),
            'total_models': len(df['model'].unique())
        }

        return InversionAnalysis(
            pairwise_stats=all_pairwise_stats,
            ranking_results=ranking_results,
            metadata=metadata
        )

    def save_results(self, analysis: InversionAnalysis, path: Union[str, Path]):
        """Save analysis results to JSON.

        Args:
            analysis: Analysis results
            path: Output file path
        """
        # Convert to serializable format
        data = {
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

            data['pairwise_stats'].append(stats_dict)

        for result in analysis.ranking_results:
            data['ranking_results'].append({
                'language': result.language,
                'model': result.model,
                'metric': result.metric,
                'mean_score': result.mean_score,
                'rank': result.rank,
                'n_conversations': result.n_conversations
            })

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)