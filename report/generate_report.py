"""Generate PDF report from training results and paper comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_METRICS = {
    "accuracy": 0.9871,
    "precision": 0.9885,
    "recall": 0.9871,
    "f1": 0.9872,
}

DATASET_BLURBS = {
    "plant_disease_expert": (
        "Plant Disease Expert (Kaggle) — the dataset used in Adnan et al. (IEEE Access 2023). "
        "Originally ~59,809 images across 58 classes; classes with fewer than 36 images are "
        "removed and each class is capped at 200 (AADL trim), yielding ~52 classes."
    ),
    "plantvillage": (
        "PlantVillage — widely used benchmark of healthy and diseased leaf images across "
        "14 crop species and 38 disease/health classes (lab-captured leaves)."
    ),
    "tea_leaf": (
        "Tea Leaf Disease — specialized multi-class tea leaf disease dataset "
        "(e.g. algal leaf, anthracnose, bird eye spot, brown blight, red leaf spot, healthy)."
    ),
}


def _load_runs(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_root.rglob("metrics.json")):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        dataset = path.parent.parent.name
        rows.append(
            {
                "dataset": dataset,
                "model": m.get("model", path.parent.name),
                "test_metrics": m.get("test_metrics", {}),
                "best_val_accuracy": m.get("best_val_accuracy"),
                "num_classes": m.get("num_classes"),
                "split": m.get("split", {}),
                "history": m.get("history"),
                "run_dir": path.parent,
                "metrics": m,
            }
        )
    return rows


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):.2f}%"


def generate_pdf(results_root: Path, out_path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    rows = _load_runs(results_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=16,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["BodyText"],
            alignment=TA_JUSTIFY,
            fontSize=10,
            leading=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Custom",
            parent=styles["Heading2"],
            fontSize=13,
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3Custom",
            parent=styles["Heading3"],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    story: List[Any] = []

    story.append(
        Paragraph(
            "Plant Disease Classification: EfficientNetB3-AADL Reproduction "
            "and AlexNet Comparison",
            styles["TitleCenter"],
        )
    )
    story.append(
        Paragraph(
            f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            "PyTorch implementation of Adnan et al., IEEE Access 2023",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    # 1. Introduction
    story.append(Paragraph("1. Introduction", styles["H2Custom"]))
    story.append(
        Paragraph(
            "This report regenerates experimental results inspired by "
            "<i>EfficientNetB3-Adaptive Augmented Deep Learning (AADL) for Multi-Class "
            "Plant Disease Classification</i> (Adnan et al., IEEE Access, 2023). "
            "The paper proposed trimming and augmenting a large multi-class leaf-disease "
            "dataset and fine-tuning a customized EfficientNetB3 model, reporting "
            f"<b>{_pct(PAPER_METRICS['accuracy'])}</b> test accuracy on 52 classes. "
            "Here we re-implement the AADL data pipeline in PyTorch, train EfficientNetB3-AADL "
            "on the paper dataset, and compare <b>AlexNet</b> versus <b>EfficientNetB3</b> "
            "across three datasets.",
            styles["BodyJust"],
        )
    )

    # 2. Datasets
    story.append(Paragraph("2. Datasets", styles["H2Custom"]))
    story.append(
        Paragraph(
            "All datasets are loaded in ImageFolder format. Classes with fewer than 36 images "
            "are dropped; each remaining class is capped at 200 images. Data are split "
            "stratified 90% / 5% / 5% (train / validation / test). Training images receive "
            "horizontal flip, rotation (±20°), translation (0.2), and zoom (~±20%). "
            "Batch size is 20.",
            styles["BodyJust"],
        )
    )
    for key, blurb in DATASET_BLURBS.items():
        story.append(Paragraph(key.replace("_", " ").title(), styles["H3Custom"]))
        story.append(Paragraph(blurb, styles["BodyJust"]))
        ds_rows = [r for r in rows if r["dataset"] == key]
        if ds_rows:
            split = ds_rows[0].get("split") or {}
            story.append(
                Paragraph(
                    f"Observed after AADL trim/split — classes: {ds_rows[0].get('num_classes')}, "
                    f"train: {split.get('train_images', '—')}, "
                    f"val: {split.get('val_images', '—')}, "
                    f"test: {split.get('test_images', '—')}.",
                    styles["BodyJust"],
                )
            )

    # 3. Methods
    story.append(Paragraph("3. Methods", styles["H2Custom"]))
    story.append(
        Paragraph(
            "<b>EfficientNetB3-AADL.</b> ImageNet-pretrained EfficientNet-B3 backbone with a "
            "custom head: Conv2d (32 filters, 2×2), MaxPool (2×2), BatchNorm, global average "
            "pool, Dense(256) with L1/L2 regularization, Dropout(0.4), and a softmax logits "
            "layer sized to the number of classes. Input resolution 300×300. Optimizer: Adamax "
            "(lr=0.001), categorical cross-entropy, 10 epochs.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>AlexNet.</b> ImageNet-pretrained AlexNet with the final fully connected layer "
            "replaced for the target class count. Input resolution 224×224. Same optimizer, "
            "loss, batch size, and epoch budget for a fair comparison.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "Code modules: <font face='Courier'>src/data.py</font> (AADL loaders), "
            "<font face='Courier'>src/models.py</font>, "
            "<font face='Courier'>src/train.py</font>, "
            "<font face='Courier'>src/evaluate.py</font>, "
            "<font face='Courier'>scripts/run_all.py</font>.",
            styles["BodyJust"],
        )
    )

    # 4. Results table
    story.append(Paragraph("4. Experimental Results", styles["H2Custom"]))

    if not rows:
        story.append(
            Paragraph(
                "<b>No trained metrics found yet.</b> Place datasets under "
                "<font face='Courier'>data/</font> and run "
                "<font face='Courier'>python scripts/run_all.py</font>, then re-run this "
                "report generator. Paper reference metrics are included below for comparison.",
                styles["BodyJust"],
            )
        )
    else:
        table_data = [
            ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1"],
        ]
        for r in rows:
            tm = r["test_metrics"]
            table_data.append(
                [
                    r["dataset"].replace("_", " "),
                    r["model"],
                    _pct(tm.get("accuracy")),
                    _pct(tm.get("precision")),
                    _pct(tm.get("recall")),
                    _pct(tm.get("f1")),
                ]
            )
        # Paper reference row
        table_data.append(
            [
                "Plant Disease Expert (paper)",
                "EfficientNetB3-AADL",
                _pct(PAPER_METRICS["accuracy"]),
                _pct(PAPER_METRICS["precision"]),
                _pct(PAPER_METRICS["recall"]),
                _pct(PAPER_METRICS["f1"]),
            ]
        )

        table = Table(table_data, colWidths=[1.5 * inch, 1.4 * inch, 0.75 * inch, 0.8 * inch, 0.7 * inch, 0.6 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#eef3f8")]),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fff2cc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "Yellow row = published paper metrics (Adnan et al., 2023). "
                "Reproduced numbers may differ due to random splits, library versions, "
                "and hardware nondeterminism.",
                styles["BodyJust"],
            )
        )

    # Paper dataset focus
    story.append(Paragraph("4.1 Paper Dataset Reproduction", styles["H3Custom"]))
    paper_runs = [
        r for r in rows if r["dataset"] == "plant_disease_expert" and "efficientnet" in r["model"]
    ]
    if paper_runs:
        tm = paper_runs[0]["test_metrics"]
        story.append(
            Paragraph(
                f"Reproduced EfficientNetB3-AADL test accuracy: <b>{_pct(tm.get('accuracy'))}</b> "
                f"(paper: {_pct(PAPER_METRICS['accuracy'])}). "
                f"Precision {_pct(tm.get('precision'))}, recall {_pct(tm.get('recall'))}, "
                f"F1 {_pct(tm.get('f1'))}.",
                styles["BodyJust"],
            )
        )
    else:
        story.append(
            Paragraph(
                "EfficientNetB3 results on Plant Disease Expert are not present in "
                "<font face='Courier'>results/</font> yet.",
                styles["BodyJust"],
            )
        )

    # 5. Comparison discussion
    story.append(Paragraph("5. AlexNet vs EfficientNetB3", styles["H2Custom"]))
    story.append(
        Paragraph(
            "Across datasets, EfficientNetB3 is expected to outperform AlexNet due to "
            "compound scaling, richer features, and the AADL-balanced training regime. "
            "AlexNet remains a useful lightweight baseline. The grouped bar charts below "
            "(when available) summarize test accuracy and F1 side by side.",
            styles["BodyJust"],
        )
    )

    # Embed comparison / curve images if present
    story.append(Paragraph("6. Figures", styles["H2Custom"]))
    image_candidates = [
        results_root / "comparison_accuracy.png",
        results_root / "comparison_f1.png",
    ]
    for r in rows:
        image_candidates.append(r["run_dir"] / "training_curves.png")
        image_candidates.append(r["run_dir"] / "confusion_matrix.png")

    embedded = 0
    for img_path in image_candidates:
        if not img_path.exists():
            continue
        story.append(Paragraph(str(img_path.relative_to(ROOT)), styles["H3Custom"]))
        # Fit width to page
        img = Image(str(img_path))
        max_w = 6.5 * inch
        max_h = 4.2 * inch
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_w / iw, max_h / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        story.append(img)
        story.append(Spacer(1, 8))
        embedded += 1
        if embedded >= 8:
            break

    if embedded == 0:
        story.append(
            Paragraph(
                "No figure PNGs found. After training, run "
                "<font face='Courier'>python src/evaluate.py</font> to create curves, "
                "confusion matrices, and comparison charts.",
                styles["BodyJust"],
            )
        )

    # 7. Conclusion
    story.append(Paragraph("7. Conclusion", styles["H2Custom"]))
    story.append(
        Paragraph(
            "This assignment re-implements the AADL data pipeline and EfficientNetB3 head from "
            "the reference paper in PyTorch, regenerates metrics on the discussed dataset, and "
            "extends the study with an AlexNet baseline on two additional plant-disease "
            "datasets. Source code, metrics JSON, and figures accompany this PDF for full "
            "reproducibility.",
            styles["BodyJust"],
        )
    )

    story.append(Paragraph("References", styles["H2Custom"]))
    story.append(
        Paragraph(
            "Adnan, F., Awan, M. J., Mahmoud, A., Nobanee, H., Yasin, A., &amp; Zain, A. M. (2023). "
            "EfficientNetB3-Adaptive Augmented Deep Learning (AADL) for Multi-Class Plant Disease "
            "Classification. <i>IEEE Access</i>, 11, 85426–85440. "
            "https://doi.org/10.1109/ACCESS.2023.3303131",
            styles["BodyJust"],
        )
    )

    doc.build(story)
    print(f"Wrote report -> {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PDF results report")
    p.add_argument(
        "--results-root",
        type=str,
        default=str(ROOT / "results"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "report" / "Plant_Disease_Results_Report.pdf"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_pdf(Path(args.results_root), Path(args.output))
