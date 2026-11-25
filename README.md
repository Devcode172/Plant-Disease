# Plant Disease Classification Assignment

PyTorch reproduction of **EfficientNetB3-AADL** (Adnan et al., IEEE Access 2023) plus an **AlexNet vs EfficientNetB3** comparison on three datasets.

## Deliverables

| Item | Location |
|------|----------|
| Code | `src/`, `scripts/` |
| Results (metrics, plots) | `results/` |
| PDF report | `report/Plant_Disease_Results_Report.pdf` |

## Setup

```bash
cd "Plant disease Assignment"
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU strongly recommended (Colab / Kaggle / local CUDA).

## Data

See [scripts/download_datasets.md](scripts/download_datasets.md). Expected layout:

```text
data/plant_disease_expert/<class>/*.jpg
data/plantvillage/<class>/*.jpg
data/tea_leaf/<class>/*.jpg
```

Optional smoke dataset:

```bash
python scripts/make_dummy_data.py
```

## Train

Single run (saves under `results/{dataset}_{model}/`):

```bash
python src/train.py --data-root data/plant_disease_expert --model efficientnet_b3 --dataset-name plant_disease_expert
python src/train.py --data-root data/plant_disease_expert --model alexnet --dataset-name plant_disease_expert
```

All dataset × model combinations (skips runs that already have `metrics.json`):

```bash
python src/run_all.py --epochs 10
```

CPU smoke test (2 classes, 20 images, 1 epoch) — run before Kaggle/GPU:

```bash
python scripts/make_dummy_data.py
python src/train.py --smoke --model alexnet --data-root data/dummy_plants
python src/run_all.py --smoke
```

## Evaluate & PDF

```bash
python src/evaluate.py --results-root results
python report/generate_report.py
```

## Pipeline (AADL)

1. ImageFolder load  
2. Drop classes with &lt; 36 images; cap at 200 / class  
3. Stratified split **90 / 5 / 5**  
4. Train aug: horizontal flip, rotation 20°, translate 0.2, zoom ±20%  
5. Batch size **20**, Adamax **lr=0.001**, **10** epochs  

Paper reference (EfficientNetB3-AADL): accuracy **98.71%**, precision 98.85%, recall 98.71%, F1 98.72%.

## Citation

Adnan et al., *EfficientNetB3-Adaptive Augmented Deep Learning (AADL) for Multi-Class Plant Disease Classification*, IEEE Access, 2023. https://doi.org/10.1109/ACCESS.2023.3303131
