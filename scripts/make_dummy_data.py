"""Create a tiny ImageFolder dataset for pipeline smoke tests (not for real metrics)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def make_class_images(class_dir: Path, n: int, seed: int, color: tuple[int, int, int]) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[:, :] = color
        noise = rng.integers(0, 40, size=(128, 128, 3), dtype=np.uint8)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(class_dir / f"img_{i:03d}.jpg")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(ROOT / "data" / "dummy_plants"))
    p.add_argument("--images-per-class", type=int, default=40)
    args = p.parse_args()

    out = Path(args.out)
    classes = {
        "healthy_leaf": (40, 140, 40),
        "leaf_blight": (140, 100, 30),
        "leaf_spot": (160, 40, 40),
        "rust": (180, 90, 40),
    }
    for i, (name, color) in enumerate(classes.items()):
        make_class_images(out / name, args.images_per_class, seed=100 + i, color=color)

    # Register as a runnable dataset key by also documenting path data/dummy_plants
    print(f"Wrote dummy dataset -> {out} ({len(classes)} classes, {args.images_per_class} images each)")
    print("Train with:")
    print(
        f'  python src/train.py --data-root "{out}" --model alexnet '
        f'--output-dir results/dummy_plants/alexnet --epochs 1'
    )


if __name__ == "__main__":
    main()
