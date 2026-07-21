# DETR3D in Pure PyTorch

This repository reproduces DETR3D without MMDetection, MMDetection3D, or MMCV.
The implementation follows the official ResNet-101 + DCNv2 + GridMask model,
training objective, data semantics, initialization, schedule, and nuScenes
evaluation behavior using PyTorch and TorchVision components.

## Reproduction Result

The faithful single-seed run was trained for 24 epochs on the official nuScenes
train split and evaluated on all 6,019 validation samples.

| Implementation | Variant | Split | mAP | NDS |
|---|---|---|---:|---:|
| Official DETR3D | ResNet-101 + DCN + GridMask | val | 34.7 | 42.2 |
| Official DETR3D | ResNet-101 + DCN + GridMask + CBGS | val | 34.9 | 43.4 |
| Official DETR3D | VoVNet + GridMask + CBGS | test | 41.2 | 47.9 |
| Pure PyTorch, earlier non-faithful run | Historical local configuration | val | 31.73 | 38.75 |
| **Pure PyTorch, faithful reproduction** | **ResNet-101 + DCN + GridMask** | **val** | **34.58** | **42.15** |

The VoVNet result uses another backbone, CBGS, trainval training data, and test
evaluation, so it is not directly comparable to the validation runs. The local
result is a successful single-seed reproduction, not a claim of bit-identical
or multi-seed statistical equivalence with upstream DDP training.

## Implementation

OpenMMLab behavior is reproduced with local modules:

- TorchVision ResNet-101 and `DeformConv2d` replace the MMDetection backbone
  and MMCV modulated DCNv2 operator.
- Local FPN, GridMask, transformer, cross-attention, detection head, losses,
  box coder, and training loop replace MMDetection/MMCV modules.
- The local nuScenes dataset and CBGS implementation replace MMDetection3D data
  wrappers and pipelines.
- SciPy provides exact Hungarian assignment.
- The official nuScenes devkit is called directly for mAP, NDS, and TP errors.

The official FCOS3D checkpoint supplies backbone/FPN initialization. Its weights
are translated into this repository's parameter names at load time; OpenMMLab
is not required at runtime.

## Repository Layout

```text
detr3d/
├── train.py                         # training entry point
├── eval.py                          # diagnostics and official evaluation
├── COMMAND_GUIDE.md                 # canonical commands
├── notebooks/
│   └── detr3d_e2e_walkthrough.ipynb
├── detr3d/
│   ├── data/                        # nuScenes data and CBGS
│   ├── engine/                      # trainer, diagnostics, evaluator
│   ├── models/                      # DETR3D architecture and checkpoint loader
│   ├── experiments/                 # reproducible search protocols
│   └── scripts/                     # regression and benchmark tools
├── docs/analysis/                   # implementation and result rationale
└── tests/
```

Datasets, checkpoints, MLflow stores, logs, and generated outputs are deliberately
excluded from version control.

## Setup

Create an environment with Python 3.11 and install the package dependencies:

```bash
pip install -r requirements.txt
```

Prepare nuScenes `v1.0-trainval` using its standard directory layout. Dataset
files are not distributed with this repository.

The faithful training path also requires the official FCOS3D initialization
checkpoint. Place it at `checkpoints/fcos3d.pth` or change the path in the
training command. The checkpoint used for the reproduced run had SHA-256:

```text
b6cd0590879adc21d2a54bebde44811391f006434a2ffcdc6b41480b2e95be48
```

## Training And Evaluation

`COMMAND_GUIDE.md` is the canonical command reference. The exact successful
C6 launch and recovery details are recorded in
`docs/analysis/c6-full-training.md`.

Useful entry points:

```bash
# Deterministic one-sample regression
python detr3d/scripts/overfit_one_batch.py --help

# Throughput and memory benchmark
python detr3d/scripts/benchmark_forward.py --help

# Full training options
python train.py --help

# Official nuScenes evaluation options
python eval.py --help
```

## Educational Walkthrough

`notebooks/detr3d_e2e_walkthrough.ipynb` explains the current package rather
than maintaining a second implementation. It covers the data contract,
LiDAR-to-image geometry, official preprocessing, model path, iterative reference
refinement, encoded boxes, matching/losses, training recipe, diagnostics, and
official evaluation.

Set `NUSCENES_DATAROOT` and optionally `DETR3D_CHECKPOINT` before running its
dataset and checkpoint sections. The notebook is intentionally committed
without execution outputs so generated data does not become repository history.

## Verification

Run the unit suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

For behavioral changes, also run the canonical one-sample regression from
`COMMAND_GUIDE.md` and compare against the recorded baseline before launching
larger experiments.
