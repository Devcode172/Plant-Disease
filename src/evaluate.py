"""Plot confusion matrices, training curves, and comparison charts from results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def plot_training_curves(history: dict, out_path: Path, title: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str = "Confusion Matrix",
    max_labels: int = 30,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    fig_w = max(8, min(n * 0.35, 24))
    fig_h = max(6, min(n * 0.35, 20))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    show_annot = n <= max_labels
    labels = class_names if n <= max_labels else None
    sns.heatmap(
        cm,
        annot=show_annot,
        fmt="d",
        cmap="Blues",
        xticklabels=labels if labels else False,
        yticklabels=labels if labels else False,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if labels:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    rows: List[Dict],
    out_path: Path,
    metric: str = "accuracy",
) -> None:
    """
    rows: list of dicts with keys dataset, model, and test_metrics[metric]
    Grouped bar chart: datasets on x-axis, AlexNet vs EfficientNet bars.
    """
    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})
    x = np.arange(len(datasets))
    width = 0.35 if len(models) <= 2 else 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            match = next(
                (r for r in rows if r["dataset"] == ds and r["model"] == model),
                None,
            )
            vals.append(match["test_metrics"][metric] if match else 0.0)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0, 1.05)
    ax.set_title(f"AlexNet vs EfficientNetB3 — Test {metric.capitalize()}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_run_dir(run_dir: Path) -> Optional[Dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    preds_path = run_dir / "predictions.npz"
    class_path = run_dir / "class_names.json"
    class_names = metrics.get("class_names")
    if class_names is None and class_path.exists():
        with open(class_path, encoding="utf-8") as f:
            class_names = json.load(f)

    if "history" in metrics:
        plot_training_curves(
            metrics["history"],
            run_dir / "training_curves.png",
            title=f"{metrics.get('model', '')} — {run_dir.name}",
        )

    if preds_path.exists() and class_names:
        data = np.load(preds_path)
        plot_confusion_matrix(
            data["y_true"],
            data["y_pred"],
            class_names,
            run_dir / "confusion_matrix.png",
            title=f"Confusion Matrix — {metrics.get('model', run_dir.name)}",
        )

    return metrics


def collect_and_compare(results_root: Path, out_dir: Optional[Path] = None) -> List[Dict]:
    out_dir = out_dir or results_root
    rows: List[Dict] = []
    for metrics_path in sorted(results_root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        metrics = process_run_dir(run_dir)
        if not metrics:
            continue
        # Infer dataset name from parent folder: results/{dataset}/{model}/
        dataset = run_dir.parent.name if run_dir.parent != results_root else run_dir.name
        rows.append(
            {
                "dataset": dataset,
                "model": metrics.get("model", run_dir.name),
                "test_metrics": metrics["test_metrics"],
                "run_dir": str(run_dir),
            }
        )

    if rows:
        plot_model_comparison(rows, out_dir / "comparison_accuracy.png", metric="accuracy")
        plot_model_comparison(rows, out_dir / "comparison_f1.png", metric="f1")
        with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote comparison charts to {out_dir}")
    else:
        print(f"No metrics.json found under {results_root}")
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / plot plant disease results")
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Single run directory containing metrics.json",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=str(ROOT / "results"),
        help="Root folder to scan for all runs and build comparison plots",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_dir:
        process_run_dir(Path(args.run_dir))
    collect_and_compare(Path(args.results_root))
