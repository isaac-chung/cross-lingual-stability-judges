"""Evaluation logic for Label Recovery predictions."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .models import ClassificationResult

console = Console()

CATEGORIES = ["industry", "problem", "channel", "agent_experience", "agent_type"]


def load_ground_truth_from_file(json_path: str | Path) -> dict[str, dict[str, str]]:
    """Load ground truth labels from a configuration JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Dictionary mapping conversation_id (as string) to ground truth labels.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = {}

    for idx, entry in enumerate(data):
        ticket_id = entry.get("ticket_id", f"id_{idx}")

        # Extract conversation ID from ticket_id (format: "id_N")
        if ticket_id.startswith("id_"):
            conversation_id = ticket_id[3:]  # Remove "id_" prefix
        else:
            # Use index for UUIDs or other formats
            conversation_id = str(idx)

        ground_truth[conversation_id] = {
            "industry": entry["config"]["industry"],
            "problem": entry["config"]["problem"],
            "channel": entry["config"]["channel"],
            "agent_experience": entry["config"]["agent_experience"],
            "agent_type": entry["config"]["agent_type"],
        }

    return ground_truth


def load_ground_truth(json_paths: list[str | Path]) -> dict[str, dict[str, str]]:
    """Load ground truth labels from multiple configuration JSON files.

    Args:
        json_paths: List of paths to JSON files.

    Returns:
        Combined dictionary mapping conversation_id to ground truth labels.
    """
    combined = {}

    for json_path in json_paths:
        path = Path(json_path)
        if path.exists():
            console.print(f"Loading ground truth from: {path.name}", style="dim")
            file_gt = load_ground_truth_from_file(path)
            console.print(f"  Loaded {len(file_gt)} conversations", style="dim")

            # Check for overlapping IDs
            overlapping = set(combined.keys()) & set(file_gt.keys())
            if overlapping:
                console.print(
                    f"  Warning: {len(overlapping)} overlapping IDs (will be overwritten)",
                    style="yellow",
                )

            combined.update(file_gt)
        else:
            console.print(f"  Warning: File not found: {json_path}", style="yellow")

    return combined


def evaluate_predictions(
    predictions: list[ClassificationResult],
    ground_truth: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Evaluate predictions against ground truth.

    Args:
        predictions: List of ClassificationResult objects.
        ground_truth: Ground truth labels dictionary.

    Returns:
        Dictionary with evaluation results per category.
    """
    results: dict[str, Any] = {
        "total_predictions": len(predictions),
        "categories": {},
    }

    for category in CATEGORIES:
        y_true = []
        y_pred = []

        for pred in predictions:
            conv_id = str(pred.conversation_id)

            # Skip if no ground truth for this conversation
            if conv_id not in ground_truth:
                continue

            # Skip if prediction is missing
            pred_value = getattr(pred, category, None)
            if pred_value is None:
                continue

            y_true.append(ground_truth[conv_id][category])
            y_pred.append(pred_value)

        if not y_true:
            results["categories"][category] = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "valid_predictions": 0,
                "unique_labels_true": 0,
                "unique_labels_pred": 0,
            }
            continue

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        results["categories"][category] = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "valid_predictions": len(y_true),
            "unique_labels_true": len(set(y_true)),
            "unique_labels_pred": len(set(y_pred)),
            "classification_report": classification_report(
                y_true, y_pred, zero_division=0, output_dict=True
            ),
        }

    return results


def display_results_table(all_results: list[dict[str, Any]]) -> None:
    """Display evaluation results in a rich table format.

    Args:
        all_results: List of evaluation results (each with 'model_name' and 'categories').
    """
    console.print("\nLabel Recovery Evaluation Results", style="bold cyan")
    console.print("=" * 60)

    table = Table(title="Model Performance Comparison", box=box.ROUNDED)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Valid Pred", justify="center", style="blue")
    table.add_column("Overall Acc", justify="center", style="green")
    table.add_column("Acc Std", justify="center", style="green")
    table.add_column("Industry F1", justify="center", style="yellow")
    table.add_column("Problem F1", justify="center", style="magenta")
    table.add_column("Channel F1", justify="center", style="red")
    table.add_column("Exp F1", justify="center", style="blue")
    table.add_column("Type F1", justify="center", style="green")

    for result in all_results:
        model_name = result.get("model_name", "unknown")
        categories = result.get("categories", {})

        # Calculate overall accuracy
        accuracies = [
            categories[cat]["accuracy"]
            for cat in CATEGORIES
            if categories.get(cat, {}).get("valid_predictions", 0) > 0
        ]
        overall_accuracy = np.mean(accuracies) if accuracies else 0.0
        accuracy_std = np.std(accuracies) if accuracies else 0.0

        # Get F1 scores
        f1_scores = {cat: categories.get(cat, {}).get("f1_macro", 0.0) for cat in CATEGORIES}

        # Get valid predictions count
        valid_preds = categories.get("industry", {}).get("valid_predictions", 0)

        table.add_row(
            model_name[:25] + "..." if len(model_name) > 25 else model_name,
            str(valid_preds),
            f"{overall_accuracy:.3f}",
            f"{accuracy_std:.3f}",
            f"{f1_scores['industry']:.3f}",
            f"{f1_scores['problem']:.3f}",
            f"{f1_scores['channel']:.3f}",
            f"{f1_scores['agent_experience']:.3f}",
            f"{f1_scores['agent_type']:.3f}",
        )

    console.print(table)


def display_detailed_results(all_results: list[dict[str, Any]]) -> None:
    """Display detailed results for each category and model.

    Args:
        all_results: List of evaluation results.
    """
    for category in CATEGORIES:
        console.print(f"\nDetailed Results for {category.upper()}", style="bold yellow")

        table = Table(title=f"{category.title()} Classification Results", box=box.SIMPLE)
        table.add_column("Model", style="cyan")
        table.add_column("Accuracy", justify="center", style="green")
        table.add_column("F1 Macro", justify="center", style="yellow")
        table.add_column("F1 Weighted", justify="center", style="blue")
        table.add_column("Valid Pred", justify="center", style="magenta")
        table.add_column("Labels (GT)", justify="center", style="red")
        table.add_column("Labels (Pred)", justify="center", style="red")

        for result in all_results:
            model_name = result.get("model_name", "unknown")
            cat_result = result.get("categories", {}).get(category, {})

            table.add_row(
                model_name[:25] + "..." if len(model_name) > 25 else model_name,
                f"{cat_result.get('accuracy', 0):.3f}",
                f"{cat_result.get('f1_macro', 0):.3f}",
                f"{cat_result.get('f1_weighted', 0):.3f}",
                str(cat_result.get("valid_predictions", 0)),
                str(cat_result.get("unique_labels_true", 0)),
                str(cat_result.get("unique_labels_pred", 0)),
            )

        console.print(table)
