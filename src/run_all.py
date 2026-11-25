"""
Loop over 3 datasets × 2 models (EfficientNetB3, AlexNet), call train,
skip runs that already have metrics.json, then write results/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import run_name, train

# Default ImageFolder roots (name, path)
DATASETS = [
    ("plant_disease_expert", ROOT / "data" / "plant_disease_expert"),
    ("plantvillage", ROOT / "data" / "plantvillage"),
    ("tea_leaf", ROOT / "data" / "tea_leaf"),
]

MODELS = ("efficientnet_b3", "alexnet")

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "training_time_seconds",
    "parameter_count",
    "num_classes",
    "epochs",
    "device",
    "run_dir",
]


def write_summary(results_root: Path, out_csv: Path) -> int:
    """Collect every results/*/metrics.json into a comparison CSV."""
    rows = []
    for metrics_path in sorted(results_root.glob("*/metrics.json")):
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        rows.append(
            {
                "dataset": m.get("dataset", metrics_path.parent.name),
                "model": m.get("model", ""),
                "accuracy": m.get("accuracy"),
                "precision_weighted": m.get("precision_weighted"),
                "recall_weighted": m.get("recall_weighted"),
                "f1_weighted": m.get("f1_weighted"),
                "training_time_seconds": m.get("training_time_seconds"),
                "parameter_count": m.get("parameter_count"),
                "num_classes": m.get("num_classes"),
                "epochs": m.get("epochs"),
                "device": m.get("device"),
                "run_dir": str(metrics_path.parent),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary ({len(rows)} runs) -> {out_csv}")
    return len(rows)


def parse_datasets(items: list[str] | None) -> list[tuple[str, Path]]:
    if not items:
        return list(DATASETS)
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
            out.append((name, Path(path)))
        else:
            path = Path(item)
            out.append((path.name, path))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train EfficientNetB3 and AlexNet on three datasets"
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help='Override datasets as paths or name=path (default: the 3 under data/)',
    )
    p.add_argument("--models", nargs="*", default=list(MODELS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--output-root", type=str, default=str(ROOT / "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-train even when metrics.json already exists",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="CPU smoke on data/dummy_plants (2 classes, 1 epoch)",
    )
    args = p.parse_args()

    output_root = Path(args.output_root)

    if args.smoke:
        dummy = ROOT / "data" / "dummy_plants"
        if not dummy.is_dir():
            import runpy

            runpy.run_path(
                str(ROOT / "scripts" / "make_dummy_data.py"), run_name="__main__"
            )
        dataset_list = [("dummy_plants", dummy)]
        smoke = True
    else:
        dataset_list = parse_datasets(args.datasets)
        smoke = False

    for ds_name, ds_path in dataset_list:
        if not ds_path.is_dir():
            print(f"[skip] missing dataset: {ds_path}")
            continue

        for model_name in args.models:
            out_dir = output_root / run_name(ds_name, model_name)
            metrics_path = out_dir / "metrics.json"

            # Resume: skip completed runs
            if metrics_path.exists() and not args.force:
                print(f"[resume-skip] {metrics_path}")
                continue

            print("\n" + "=" * 72)
            print(f"Training {model_name} on {ds_name} ({ds_path})")
            print("=" * 72)
            train(
                data_root=ds_path,
                model_name=model_name,
                dataset_name=ds_name,
                output_root=output_root,
                epochs=1 if smoke else args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                pretrained=False if smoke else (not args.no_pretrained),
                seed=args.seed,
                smoke=smoke,
            )

    write_summary(output_root, output_root / "summary.csv")


if __name__ == "__main__":
    main()
