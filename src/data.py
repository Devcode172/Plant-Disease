"""AADL-style ImageFolder loading, class trim, stratified split, and DataLoaders."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePathDataset(Dataset):
    """Simple dataset of (image_path, label) pairs with a transform."""

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _collect_imagefolder(root: str | Path) -> Tuple[dict[str, List[str]], List[str]]:
    """Walk ImageFolder layout and return {class_name: [paths]} and sorted class names."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_to_paths: dict[str, List[str]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.bmp", "*.webp"):
            paths.extend(str(p) for p in entry.glob(ext))
        # also search one nested level (common Kaggle layout)
        if not paths:
            for sub in entry.iterdir():
                if sub.is_dir():
                    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                        paths.extend(str(p) for p in sub.glob(ext))
        if paths:
            class_to_paths[entry.name] = sorted(paths)

    if not class_to_paths:
        raise RuntimeError(
            f"No class folders with images found under {root}. "
            "Expected ImageFolder layout: root/class_name/*.jpg"
        )

    class_names = sorted(class_to_paths.keys())
    return class_to_paths, class_names


def trim_classes(
    class_to_paths: dict[str, List[str]],
    min_images: int = 36,
    max_images: int = 200,
    max_classes: Optional[int] = None,
    seed: int = 42,
) -> Tuple[dict[str, List[str]], List[str]]:
    """Drop classes with fewer than min_images; cap each class at max_images."""
    rng = random.Random(seed)
    trimmed: dict[str, List[str]] = {}
    for name, paths in class_to_paths.items():
        if len(paths) < min_images:
            continue
        chosen = list(paths)
        if len(chosen) > max_images:
            chosen = rng.sample(chosen, max_images)
        trimmed[name] = sorted(chosen)

    class_names = sorted(trimmed.keys())
    if max_classes is not None and len(class_names) > max_classes:
        class_names = sorted(rng.sample(class_names, max_classes))
        trimmed = {n: trimmed[n] for n in class_names}

    if not class_names:
        raise RuntimeError(
            f"No classes remained after trimming (min={min_images}, max={max_images})."
        )
    return trimmed, class_names


def stratified_split(
    class_to_paths: dict[str, List[str]],
    class_names: Sequence[str],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Stratified 90/5/5 split matching the paper:
    90% train, remaining 10% split evenly into val and test.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    paths: List[str] = []
    labels: List[int] = []
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for name in class_names:
        for p in class_to_paths[name]:
            paths.append(p)
            labels.append(name_to_idx[name])

    train_paths, hold_paths, train_labels, hold_labels = train_test_split(
        paths,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Split hold-out evenly into val / test (paper: remaining 10% half/half)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        hold_paths,
        hold_labels,
        test_size=relative_test,
        stratify=hold_labels,
        random_state=seed,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))
    test_samples = list(zip(test_paths, test_labels))
    return train_samples, val_samples, test_samples


def build_transforms(
    image_size: int,
    train: bool = True,
) -> transforms.Compose:
    """
    Train: resize, horizontal flip, rotation 20, translate 0.2, zoom 0.2.
    Val/Test: resize only (+ normalize).
    """
    if train:
        # scale range for ~20% zoom: [1-0.2, 1+0.2] -> torchvision uses scale as area factor
        # RandomAffine scale is relative to original size: (min, max)
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),  # zoom ~±20%
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_dataloaders(
    root: str | Path,
    image_size: int = 224,
    batch_size: int = 20,
    min_images: int = 36,
    max_images: int = 200,
    max_classes: Optional[int] = None,
    num_workers: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Load ImageFolder-style data, trim classes, stratified 90/5/5 split,
    apply AADL-style train augmentation, and return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    class_to_paths, _ = _collect_imagefolder(root)
    trimmed, class_names = trim_classes(
        class_to_paths,
        min_images=min_images,
        max_images=max_images,
        max_classes=max_classes,
        seed=seed,
    )
    train_samples, val_samples, test_samples = stratified_split(
        trimmed, class_names, seed=seed
    )

    train_ds = ImagePathDataset(train_samples, transform=build_transforms(image_size, train=True))
    val_ds = ImagePathDataset(val_samples, transform=build_transforms(image_size, train=False))
    test_ds = ImagePathDataset(test_samples, transform=build_transforms(image_size, train=False))

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 0)

    pin = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    return train_loader, val_loader, test_loader, class_names


def summarize_split(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: Sequence[str],
) -> dict:
    """Return a small summary dict useful for logging / reports."""
    return {
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "train_images": len(train_loader.dataset),
        "val_images": len(val_loader.dataset),
        "test_images": len(test_loader.dataset),
        "batch_size": train_loader.batch_size,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Smoke-test data pipeline")
    parser.add_argument("--root", required=True, help="ImageFolder root")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    loaders = create_dataloaders(args.root, image_size=args.image_size, batch_size=args.batch_size)
    summary = summarize_split(*loaders)
    print(json.dumps({k: v for k, v in summary.items() if k != "class_names"}, indent=2))
    print(f"classes ({summary['num_classes']}): {summary['class_names'][:5]}...")
