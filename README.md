# DETR3D Pure PyTorch Scaffold

This repository now has a clean package layout for building DETR3D from scratch in pure PyTorch while keeping the original flat scratch files intact.

## Project Layout

```text
detr3d/
├── README.md
├── pyproject.toml
├── requirements.txt
├── train.py
├── eval.py
├── infer.py
├── configs/
├── notebooks/
│   └── detr3d_e2e_walkthrough.ipynb
├── detr3d/
│   ├── __init__.py
│   ├── data/
│   ├── engine/
│   ├── models/
│   │   ├── backbone/
│   │   ├── neck/
│   │   ├── transformer/
│   │   ├── heads/
│   │   └── losses/
│   ├── scripts/
│   └── utils/
├── tests/
└── outputs/
```

## Notebook Flow

Use [notebooks/detr3d_e2e_walkthrough.ipynb](/home/user/workspace/ML_study/implementations/detr3d/notebooks/detr3d_e2e_walkthrough.ipynb) as the single end-to-end notebook.

It is organized as:

1. Data preparation
2. Architecture
3. Loss and matching
4. Training loop
5. Experiment tracking and analysis

Inside the architecture section, each major component is isolated to its own cell so you can study the model incrementally.

## Current State

- The new tree is a scaffold for the cleaned project layout.
- Existing scratch implementations at the repo root are still present.
- The next practical step is to migrate real implementations into the new package modules one by one.
