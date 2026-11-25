"""
Train with Adamax (lr=0.001), 10 epochs, cross-entropy, mixed precision.
Saves to results/{dataset}_{model}/: metrics.json, curves.png,
confusion_matrix.png, model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adamax
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import create_dataloaders, summarize_split
from src.models import build_model, default_image_size


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_name(dataset: str, model_name: str) -> str:
    return f"{dataset}_{model_name}"


def _make_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds_all: List[int] = []
    labels_all: List[int] = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, labels).item()
                n_batches += 1
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(preds_all)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "loss": float(total_loss / max(n_batches, 1)),
    }
    return metrics, y_true, y_pred


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # L1/L2 outside autocast (fp32)
        if hasattr(model, "regularization_loss"):
            loss = loss + model.regularization_loss()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / max(len(loader), 1),
        "accuracy": correct / max(total, 1),
    }


def plot_curves(history: Dict[str, List[float]], out_path: Path, title: str) -> None:
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

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, min(n * 0.4, 20)), max(5, min(n * 0.4, 16))))
    annot = n <= 30
    sns.heatmap(
        cm,
        annot=annot,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if annot else False,
        yticklabels=class_names if annot else False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if annot:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(
    data_root: str | Path,
    model_name: str,
    dataset_name: Optional[str] = None,
    output_root: str | Path = ROOT / "results",
    epochs: int = 10,
    batch_size: int = 20,
    lr: float = 0.001,
    image_size: Optional[int] = None,
    pretrained: bool = True,
    seed: int = 42,
    num_workers: Optional[int] = None,
    device: Optional[str] = None,
    min_images: int = 36,
    max_images: int = 200,
    max_classes: Optional[int] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    set_seed(seed)
    data_root = Path(data_root)
    dataset_name = dataset_name or data_root.name
    out_dir = Path(output_root) / run_name(dataset_name, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        epochs = 1
        min_images = 1
        max_images = 20
        max_classes = 2
        pretrained = False
        num_workers = 0
        image_size = image_size or 128
        device_obj = torch.device("cpu")
        print("[smoke] 2 classes, max 20 images/class, 1 epoch, CPU")
    else:
        image_size = image_size or default_image_size(model_name)
        if device is None:
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

    use_amp = device_obj.type == "cuda"
    print(
        f"Device: {device_obj} | AMP={use_amp} | model={model_name} | "
        f"image_size={image_size} | epochs={epochs} | lr={lr}"
    )

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        min_images=min_images,
        max_images=max_images,
        max_classes=max_classes,
        num_workers=num_workers,
        seed=seed,
    )
    split_info = summarize_split(train_loader, val_loader, test_loader, class_names)
    print(
        f"Classes={split_info['num_classes']} | "
        f"train={split_info['train_images']} "
        f"val={split_info['val_images']} "
        f"test={split_info['test_images']}"
    )

    model = build_model(model_name, num_classes=len(class_names), pretrained=pretrained)
    model = model.to(device_obj)

    # Materialize LazyLinear (EfficientNet head) before counting params
    with torch.no_grad():
        model.eval()
        _ = model(torch.zeros(1, 3, image_size, image_size, device=device_obj))
        model.train()
    n_params = count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adamax(model.parameters(), lr=lr)
    scaler = _make_scaler(use_amp)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, device_obj, optimizer, criterion, scaler, use_amp
        )
        val_metrics, _, _ = evaluate(
            model, val_loader, device_obj, criterion, use_amp=use_amp
        )

        history["train_loss"].append(train_stats["loss"])
        history["train_acc"].append(train_stats["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
    training_seconds = time.time() - t0

    test_metrics, y_true, y_pred = evaluate(
        model, test_loader, device_obj, criterion, use_amp=use_amp
    )

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": model_name,
            "num_classes": len(class_names),
            "class_names": class_names,
            "image_size": image_size,
            "dataset": dataset_name,
        },
        out_dir / "model.pt",
    )
    plot_curves(history, out_dir / "curves.png", f"{dataset_name} / {model_name}")
    plot_confusion(
        y_true,
        y_pred,
        class_names,
        out_dir / "confusion_matrix.png",
        f"Confusion — {dataset_name} / {model_name}",
    )

    metrics = {
        "dataset": dataset_name,
        "model": model_name,
        "data_root": str(data_root.resolve()),
        "device": str(device_obj),
        "amp": use_amp,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "accuracy": test_metrics["accuracy"],
        "precision_weighted": test_metrics["precision"],
        "recall_weighted": test_metrics["recall"],
        "f1_weighted": test_metrics["f1"],
        "training_time_seconds": training_seconds,
        "parameter_count": n_params,
        "test_loss": test_metrics["loss"],
        "split": {k: v for k, v in split_info.items() if k != "class_names"},
        "history": history,
        "smoke": smoke,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"Test acc={test_metrics['accuracy']:.4f} "
        f"P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} "
        f"F1={test_metrics['f1']:.4f} | "
        f"params={n_params:,} | time={training_seconds:.1f}s"
    )
    print(f"Saved -> {out_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train plant disease classifier")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument(
        "--model",
        type=str,
        default="alexnet",
        choices=["alexnet", "efficientnet_b3"],
    )
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--output-root", type=str, default=str(ROOT / "results"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2 classes, 20 images each, 1 epoch, CPU",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root or str(ROOT / "data" / "dummy_plants"),
        model_name=args.model,
        dataset_name=args.dataset_name,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        smoke=args.smoke,
    )
