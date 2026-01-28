"""Core analysis logic for comparing judge models."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table


class JudgeAblationAnalyzer:
    """Analyzer for comparing label recovery results across judge models."""

    CATEGORIES = ["industry", "problem", "channel", "agent_experience", "agent_type"]

    def __init__(self) -> None:
        """Initialize the analyzer."""
        self.console = Console()

    def load_results(self, paths: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Load label_recovery JSONL files and group by judge model.

        Args:
            paths: List of file paths to JSONL files.

        Returns:
            Dict mapping judge model name to list of result dicts.
        """
        results_by_judge: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for path in paths:
            path_obj = Path(path)
            if not path_obj.exists():
                self.console.print(f"[yellow]Warning: File not found: {path}[/yellow]")
                continue

            with open(path_obj, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result = json.loads(line)
                    judge_model = self._get_judge_model(result, path_obj.name)
                    results_by_judge[judge_model].append(result)

        return dict(results_by_judge)

    def _get_judge_model(self, result: dict[str, Any], filename: str) -> str:
        """Get judge model from result dict or filename.

        Priority:
        1. result["judge_model"] field (if present)
        2. Parse from filename: label_recovery_{model}_{lang}_{datetime}.jsonl

        Args:
            result: Result dict from JSONL.
            filename: Filename of the JSONL file.

        Returns:
            Judge model name.
        """
        # Priority 1: judge_model field
        if "judge_model" in result and result["judge_model"] != "unknown":
            return result["judge_model"]

        # Priority 2: Parse from filename
        # Pattern: label_recovery_{model}_{lang}_{datetime}.jsonl
        pattern = r"label_recovery_(.+?)_([a-z]{2})_\d{8}-\d{6}\.jsonl"
        match = re.match(pattern, filename)
        if match:
            return match.group(1)

        return "unknown"

    def load_ground_truth(
        self, json_paths: list[str]
    ) -> dict[tuple[str, int], dict[str, str]]:
        """Load ground truth from config JSON files.

        The config JSON format is expected to have conversations grouped by model:
        {
            "model_name": [
                {"industry": "...", "problem": "...", ...},
                ...
            ]
        }

        Args:
            json_paths: Paths to ground truth JSON files.

        Returns:
            Dict mapping (generator_model, conversation_id) to ground truth labels.
        """
        ground_truth: dict[tuple[str, int], dict[str, str]] = {}

        for path in json_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                msg = f"[yellow]Warning: Ground truth file not found: {path}[/yellow]"
                self.console.print(msg)
                continue

            with open(path_obj, encoding="utf-8") as f:
                data = json.load(f)

            # Handle grouped format (by generator model)
            for generator_model, conversations in data.items():
                for idx, conv in enumerate(conversations):
                    key = (generator_model, idx)
                    ground_truth[key] = {
                        "industry": conv.get("industry", ""),
                        "problem": conv.get("problem", ""),
                        "channel": conv.get("channel", ""),
                        "agent_experience": conv.get("agent_experience", ""),
                        "agent_type": conv.get("agent_type", ""),
                    }

        return ground_truth

    def evaluate_predictions(
        self,
        results: list[dict[str, Any]],
        ground_truth: dict[tuple[str, int], dict[str, str]],
    ) -> dict[str, Any]:
        """Calculate accuracy per category for a set of predictions.

        Args:
            results: List of prediction result dicts.
            ground_truth: Dict mapping (generator_model, conv_id) to ground truth.

        Returns:
            Dict with accuracy metrics per category and overall.
        """
        category_correct: dict[str, int] = defaultdict(int)
        category_total: dict[str, int] = defaultdict(int)
        all_correct = 0
        total = 0

        for result in results:
            key = (result["generator_model"], result["conversation_id"])
            if key not in ground_truth:
                continue

            gt = ground_truth[key]
            total += 1
            all_match = True

            for category in self.CATEGORIES:
                pred_value = result.get(category, "")
                gt_value = gt.get(category, "")

                category_total[category] += 1
                if pred_value == gt_value:
                    category_correct[category] += 1
                else:
                    all_match = False

            if all_match:
                all_correct += 1

        eval_result: dict[str, Any] = {
            "total": total,
            "all_correct": all_correct,
            "all_accuracy": all_correct / total if total > 0 else 0.0,
            "category_accuracy": {},
        }

        for category in self.CATEGORIES:
            cat_total = category_total[category]
            cat_correct = category_correct[category]
            eval_result["category_accuracy"][category] = {
                "correct": cat_correct,
                "total": cat_total,
                "accuracy": cat_correct / cat_total if cat_total > 0 else 0.0,
            }

        return eval_result

    def calculate_judge_consistency(
        self,
        results_by_judge: dict[str, list[dict[str, Any]]],
        ground_truth: dict[tuple[str, int], dict[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """Calculate consistency metrics for each judge across generator models.

        For each judge, calculates std dev of accuracy across generator models.
        Lower std = more consistent performance.

        Args:
            results_by_judge: Results grouped by judge model.
            ground_truth: Ground truth labels.

        Returns:
            Dict mapping judge model to consistency metrics.
        """
        consistency: dict[str, dict[str, Any]] = {}

        for judge_model, results in results_by_judge.items():
            # Group by generator model
            by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in results:
                by_generator[r["generator_model"]].append(r)

            # Calculate accuracy per generator model
            generator_accuracies: list[float] = []
            generator_results: dict[str, dict[str, Any]] = {}

            for gen_model, gen_results in by_generator.items():
                eval_result = self.evaluate_predictions(gen_results, ground_truth)
                generator_results[gen_model] = eval_result
                generator_accuracies.append(eval_result["all_accuracy"])

            # Calculate consistency metrics
            if len(generator_accuracies) > 1:
                accuracy_std = float(np.std(generator_accuracies))
                accuracy_mean = float(np.mean(generator_accuracies))
            else:
                accuracy_std = 0.0
                accuracy_mean = generator_accuracies[0] if generator_accuracies else 0.0

            consistency[judge_model] = {
                "mean_accuracy": accuracy_mean,
                "accuracy_std": accuracy_std,
                "n_generator_models": len(by_generator),
                "generator_results": generator_results,
            }

        return consistency

    def calculate_inter_judge_agreement(
        self,
        results_by_judge: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Calculate agreement between judges on the same conversations.

        Args:
            results_by_judge: Results grouped by judge model.

        Returns:
            Dict with agreement metrics between judge pairs.
        """
        # Build lookup: (generator_model, conv_id) -> {judge_model: predictions}
        PredType = dict[tuple[str, int], dict[str, dict[str, str]]]
        predictions: PredType = defaultdict(dict)

        for judge_model, results in results_by_judge.items():
            for r in results:
                key = (r["generator_model"], r["conversation_id"])
                predictions[key][judge_model] = {
                    cat: r.get(cat, "") for cat in self.CATEGORIES
                }

        # Calculate pairwise agreement
        judge_models = list(results_by_judge.keys())
        pairwise_agreement: dict[str, dict[str, float]] = {}
        category_agreement: dict[str, list[float]] = defaultdict(list)

        for i, judge_a in enumerate(judge_models):
            for judge_b in judge_models[i + 1:]:
                pair_key = f"{judge_a} vs {judge_b}"
                n_common = 0
                n_agree = 0
                cat_agree: dict[str, int] = defaultdict(int)
                cat_total: dict[str, int] = defaultdict(int)

                for key, judge_preds in predictions.items():
                    if judge_a in judge_preds and judge_b in judge_preds:
                        n_common += 1
                        pred_a = judge_preds[judge_a]
                        pred_b = judge_preds[judge_b]

                        all_match = True
                        for cat in self.CATEGORIES:
                            cat_total[cat] += 1
                            if pred_a.get(cat, "") == pred_b.get(cat, ""):
                                cat_agree[cat] += 1
                            else:
                                all_match = False

                        if all_match:
                            n_agree += 1

                if n_common > 0:
                    pairwise_agreement[pair_key] = {
                        "agreement": n_agree / n_common,
                        "n_common": n_common,
                        "category_agreement": {
                            cat: cat_agree[cat] / cat_total[cat]
                            for cat in self.CATEGORIES
                        },
                    }

                    for cat in self.CATEGORIES:
                        category_agreement[cat].append(cat_agree[cat] / cat_total[cat])

        # Calculate overall averages
        avg_category_agreement = {
            cat: float(np.mean(agreements)) if agreements else 0.0
            for cat, agreements in category_agreement.items()
        }

        all_agreements = [
            p["agreement"] for p in pairwise_agreement.values()
        ]
        if all_agreements:
            avg_overall_agreement = float(np.mean(all_agreements))
        else:
            avg_overall_agreement = 0.0

        return {
            "pairwise": pairwise_agreement,
            "avg_overall_agreement": avg_overall_agreement,
            "avg_category_agreement": avg_category_agreement,
        }

    def display_comparison_table(
        self,
        consistency: dict[str, dict[str, Any]],
    ) -> None:
        """Display Rich table comparing judges.

        Args:
            consistency: Dict from calculate_judge_consistency().
        """
        table = Table(title="Judge Model Comparison")

        table.add_column("Judge Model", justify="left", style="cyan", no_wrap=True)
        table.add_column("Avg Accuracy", justify="center", style="green")
        table.add_column("Accuracy Std", justify="center", style="yellow")
        table.add_column("Models Evaluated", justify="center", style="blue")

        for judge_model in sorted(consistency.keys()):
            metrics = consistency[judge_model]
            table.add_row(
                judge_model,
                f"{metrics['mean_accuracy']:.1%}",
                f"{metrics['accuracy_std']:.3f}",
                str(metrics["n_generator_models"]),
            )

        self.console.print(table)

    def display_detailed_results(
        self,
        consistency: dict[str, dict[str, Any]],
        ground_truth: dict[tuple[str, int], dict[str, str]],
        results_by_judge: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Display detailed per-category results.

        Args:
            consistency: Dict from calculate_judge_consistency().
            ground_truth: Ground truth labels.
            results_by_judge: Results grouped by judge model.
        """
        for category in self.CATEGORIES:
            table = Table(title=f"Category: {category}")

            table.add_column("Judge Model", justify="left", style="cyan", no_wrap=True)
            table.add_column("Mean Accuracy", justify="center", style="green")
            table.add_column("Accuracy Std", justify="center", style="yellow")
            table.add_column("Models", justify="center", style="blue")

            for judge_model in sorted(consistency.keys()):
                # Calculate per-generator accuracy for this category
                results = results_by_judge[judge_model]
                by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for r in results:
                    by_generator[r["generator_model"]].append(r)

                category_accuracies: list[float] = []
                for gen_model, gen_results in by_generator.items():
                    correct = 0
                    total = 0
                    for r in gen_results:
                        key = (r["generator_model"], r["conversation_id"])
                        if key in ground_truth:
                            total += 1
                            pred_val = r.get(category, "")
                            gt_val = ground_truth[key].get(category, "")
                            if pred_val == gt_val:
                                correct += 1
                    if total > 0:
                        category_accuracies.append(correct / total)

                if category_accuracies:
                    mean_acc = float(np.mean(category_accuracies))
                    if len(category_accuracies) > 1:
                        std_acc = float(np.std(category_accuracies))
                    else:
                        std_acc = 0.0
                else:
                    mean_acc = 0.0
                    std_acc = 0.0

                table.add_row(
                    judge_model,
                    f"{mean_acc:.1%}",
                    f"{std_acc:.3f}",
                    str(len(by_generator)),
                )

            self.console.print(table)
            self.console.print()

    def display_agreement_summary(
        self,
        agreement: dict[str, Any],
    ) -> None:
        """Display inter-judge agreement summary.

        Args:
            agreement: Dict from calculate_inter_judge_agreement().
        """
        self.console.print("\n[bold]Inter-Judge Agreement Summary[/bold]")
        avg = agreement['avg_overall_agreement']
        self.console.print(f"Average Overall Agreement: {avg:.1%}")
        self.console.print("\nAverage Category Agreement:")
        for cat, acc in agreement["avg_category_agreement"].items():
            self.console.print(f"  {cat}: {acc:.1%}")

        if agreement["pairwise"]:
            table = Table(title="Pairwise Agreement")
            table.add_column("Judge Pair", justify="left", style="cyan")
            table.add_column("Agreement", justify="center", style="green")
            table.add_column("N Common", justify="center", style="blue")

            for pair, metrics in sorted(agreement["pairwise"].items()):
                table.add_row(
                    pair,
                    f"{metrics['agreement']:.1%}",
                    str(metrics["n_common"]),
                )

            self.console.print(table)

    def run_analysis(
        self,
        result_paths: list[str],
        ground_truth_path: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Run full analysis and display results.

        Args:
            result_paths: Paths to label_recovery JSONL files.
            ground_truth_path: Path to ground truth JSON file.
            output_path: Optional path to save detailed results as JSON.

        Returns:
            Dict with all analysis results.
        """
        # Load data
        results_by_judge = self.load_results(result_paths)
        if not results_by_judge:
            self.console.print("[red]Error: No results loaded[/red]")
            return {}

        n_judges = len(results_by_judge)
        self.console.print(f"Loaded results from {n_judges} judge model(s)")
        for judge, results in results_by_judge.items():
            self.console.print(f"  {judge}: {len(results)} results")

        ground_truth = self.load_ground_truth([ground_truth_path])
        if not ground_truth:
            self.console.print("[red]Error: No ground truth loaded[/red]")
            return {}

        self.console.print(f"Loaded {len(ground_truth)} ground truth entries")

        # Calculate metrics
        consistency = self.calculate_judge_consistency(results_by_judge, ground_truth)
        agreement = self.calculate_inter_judge_agreement(results_by_judge)

        # Display results
        self.console.print()
        self.display_comparison_table(consistency)
        self.console.print()
        self.display_detailed_results(consistency, ground_truth, results_by_judge)
        self.display_agreement_summary(agreement)

        # Build output
        output = {
            "consistency": consistency,
            "agreement": agreement,
        }

        # Save if output path specified
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            self.console.print(f"\nWrote detailed results to {output_path}")

        return output

    def display_saved_results(self, json_path: str) -> None:
        """Load and display saved analysis results.

        Args:
            json_path: Path to saved analysis JSON file.
        """
        path_obj = Path(json_path)
        if not path_obj.exists():
            self.console.print(f"[red]Error: File not found: {json_path}[/red]")
            return

        with open(path_obj, encoding="utf-8") as f:
            data = json.load(f)

        consistency = data.get("consistency", {})
        agreement = data.get("agreement", {})

        if not consistency:
            self.console.print("[red]Error: No consistency data in file[/red]")
            return

        self.console.print(f"Loaded saved results from {json_path}")

        # Display comparison table
        self.console.print()
        self.display_comparison_table(consistency)

        # Display per-category results from saved data
        self.console.print()
        self._display_saved_category_results(consistency)

        # Display agreement summary
        if agreement:
            self.display_agreement_summary(agreement)

    def _display_saved_category_results(
        self,
        consistency: dict[str, dict[str, Any]],
    ) -> None:
        """Display per-category results from saved consistency data.

        Args:
            consistency: Saved consistency dict with generator_results.
        """
        for category in self.CATEGORIES:
            table = Table(title=f"Category: {category}")

            table.add_column("Judge Model", justify="left", style="cyan", no_wrap=True)
            table.add_column("Mean Accuracy", justify="center", style="green")
            table.add_column("Accuracy Std", justify="center", style="yellow")
            table.add_column("Models", justify="center", style="blue")

            for judge_model in sorted(consistency.keys()):
                judge_data = consistency[judge_model]
                gen_results = judge_data.get("generator_results", {})

                # Extract category accuracy from each generator model
                category_accuracies: list[float] = []
                for _, eval_result in gen_results.items():
                    cat_acc = eval_result.get("category_accuracy", {})
                    if category in cat_acc:
                        category_accuracies.append(cat_acc[category]["accuracy"])

                if category_accuracies:
                    mean_acc = float(np.mean(category_accuracies))
                    if len(category_accuracies) > 1:
                        std_acc = float(np.std(category_accuracies))
                    else:
                        std_acc = 0.0
                else:
                    mean_acc = 0.0
                    std_acc = 0.0

                table.add_row(
                    judge_model,
                    f"{mean_acc:.1%}",
                    f"{std_acc:.3f}",
                    str(len(gen_results)),
                )

            self.console.print(table)
            self.console.print()
