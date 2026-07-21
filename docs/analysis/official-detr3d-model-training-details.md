# Official DETR3D Model And Training Details

## Scope And Sources

This note records the upstream ResNet-101 GridMask configuration rather than
paper-level approximations. The primary source is the official repository at
commit `34a47673011fe13593a3e594a376668acca8bddb`:

- [ResNet-101 GridMask config](https://github.com/WangYueFt/detr3d/blob/34a47673011fe13593a3e594a376668acca8bddb/projects/configs/detr3d/detr3d_res101_gridmask.py)
- [CBGS variant](https://github.com/WangYueFt/detr3d/blob/34a47673011fe13593a3e594a376668acca8bddb/projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)
- [Distributed launcher](https://github.com/WangYueFt/detr3d/blob/34a47673011fe13593a3e594a376668acca8bddb/tools/dist_train.sh)
- [Reported model results](https://github.com/WangYueFt/detr3d/blob/34a47673011fe13593a3e594a376668acca8bddb/README.md)
- [Paper implementation details](https://arxiv.org/abs/2110.06922)

The released config fixes the per-GPU batch but leaves GPU count to the
launcher. The paper removes that ambiguity for its reported recipe: eight RTX
3090 GPUs, one sample per GPU, 12 epochs, and approximately 18 hours of
training. The later released config differs from the paper in optimizer
hyperparameters and schedule, so both recipes are recorded below.

## Distributed Batch Semantics

Official config values:

- Samples per GPU: 1.
- Gradient accumulation: absent.
- Accumulation steps: 1.
- Distributed implementation: one process per GPU with synchronized DDP.
- Effective global batch formula: `samples_per_gpu * world_size * accumulation_steps`.
- Effective global batch for this config: `world_size`.

| GPUs | Samples/GPU | Accumulation | Effective global batch |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | 1 | 1 | 2 |
| 4 | 1 | 1 | 4 |
| 8 | 1 | 1 | 8 |

`optimizer_config` contains gradient clipping only. It does not configure
MMCV cumulative iterations or any other gradient-accumulation mechanism.
Every distributed iteration therefore performs an optimizer update.

For a single-GPU reproduction, matching the paper's eight-GPU global batch
requires an equivalent local configuration such as physical batch 4 with two
accumulation steps. The full-training branch implements this option and steps
the optimizer, gradient clipping, and LR scheduler only after each complete
accumulation window.

## Paper And Released-Config Recipes

| Item | Paper recipe | Released config |
|---|---:|---:|
| GPUs | 8 RTX 3090 | Set by launcher |
| Batch per GPU | 1 | 1 |
| Effective global batch | 8 | GPU count |
| Gradient accumulation | Not used or reported | Not configured |
| Epochs | 12 | 24 |
| AdamW learning rate | `1e-4` | `2e-4` |
| Weight decay | `1e-4` | `0.01` |
| LR schedule | Step decay | Cosine annealing |
| LR milestones | Epochs 8 and 11 | Not applicable |
| Warmup | Not reported | 500 iterations, ratio 1/3 |
| Training time | Approximately 18 hours | Not reported |

The paper reduces learning rate to `1e-5` at epoch 8 and `1e-6` at epoch 11.
This project follows the released-config recipe when it conflicts with the
paper, consistent with the repository reproduction policy.

## Model Configuration

| Item | Official value |
|---|---|
| Detector | `Detr3D` |
| Input modality | Six cameras only |
| Classes | 10 nuScenes detection classes |
| Object queries | 900 |
| Iterative box refinement | Enabled |
| Two-stage proposals | Disabled |
| Embedding dimension | 256 |
| Decoder layers | 6 |
| Self-attention heads | 8 |
| Attention dropout | 0.1 |
| FFN channels | 512 |
| FFN dropout | 0.1 |
| Cross-attention points | 1 |
| Decoder order | Self-attention, norm, cross-attention, norm, FFN, norm |
| Positional encoding | Sine, 128 features, normalized, offset -0.5 |

### Image Backbone

- ResNet-101 with four stages.
- Caffe convolution style.
- Stage 1 frozen through `frozen_stages=1`.
- Batch normalization parameters frozen and BN kept in evaluation mode.
- Modulated DCNv2 in stages 3 and 4.
- One deformable group.
- FCOS3D checkpoint loaded before DETR3D training.

### Feature Pyramid

- Inputs: ResNet channels 256, 512, 1024, and 2048.
- Start level: 1, so the FPN consumes 512, 1024, and 2048 channels.
- Output channels: 256.
- Number of outputs: 4.
- Extra convolutions added on the output.
- ReLU applied before extra convolutions.

### Spatial Configuration

- Point-cloud range: `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`.
- Voxel size used by the coder: `[0.2, 0.2, 8.0]`.
- Assigner grid size: `[512, 512, 1]`.
- Assigner output-size factor: 4.
- Decode post-center range: `[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]`.
- Maximum decoded predictions: 300.

## Loss And Matching

| Item | Official value |
|---|---:|
| Classification loss | Sigmoid focal loss |
| Focal alpha | 0.25 |
| Focal gamma | 2.0 |
| Classification loss weight | 2.0 |
| Bbox L1 loss weight | 0.25 |
| IoU loss weight | 0.0 |
| Hungarian classification cost | 2.0 |
| Hungarian bbox cost | 0.25 |
| Hungarian IoU cost | 0.0 |
| Velocity dimensions in matching | Excluded |
| Velocity dimensions in bbox loss | Included with code weight 0.2 each |

The official matcher compares the first eight encoded box dimensions. The two
velocity dimensions remain part of the regression output and loss but do not
affect Hungarian assignment.

## Data Pipeline

### Training

- nuScenes train split with `use_valid_flag=True`.
- Load all multiview camera images as float32.
- Apply multiview photometric distortion.
- Apply GridMask before the image backbone.
- Load 3D boxes and class labels without attribute labels.
- Filter objects by the configured point-cloud range and class list.
- Normalize in BGR order with mean `[103.530, 116.280, 123.675]`.
- Use standard deviation `[1.0, 1.0, 1.0]` and `to_rgb=False`.
- Pad images to a size divisible by 32.

### Validation And Test

- No GridMask or photometric distortion.
- Use the same BGR normalization and divisor-32 padding.
- Disable flipping.
- Use the official nuScenes evaluator for mAP, NDS, and true-positive errors.

## Optimizer And Schedule

| Item | Official value |
|---|---|
| Optimizer | AdamW |
| Base learning rate | `2e-4` |
| Backbone LR multiplier | 0.1 |
| Backbone learning rate | `2e-5` |
| Weight decay | 0.01 |
| Gradient clipping | L2 norm, maximum 35 |
| LR policy | Cosine annealing |
| Warmup | Linear for 500 iterations |
| Warmup starting ratio | 1/3 |
| Minimum LR ratio | `1e-3` |
| Total epochs | 24 |
| Evaluation interval | Every 2 epochs |
| AMP/fp16 | Not enabled by this config |
| Gradient accumulation | Not enabled |

The 500-step warmup and cosine schedule are iteration-based. Their behavior in
terms of samples processed depends on world size and effective global batch.
Changing GPU count or adding accumulation without accounting for scheduler
steps does not exactly reproduce the distributed optimization trajectory.

## Base And CBGS Results

The upstream README reports:

| Variant | mAP | NDS |
|---|---:|---:|
| ResNet-101 with DCN and GridMask | 34.7 | 42.2 |
| ResNet-101 with DCN, GridMask, and CBGS | 34.9 | 43.4 |

CBGS wraps the training dataset with class-balanced group sampling. It is a
separate official variant, not part of the base `detr3d_res101_gridmask.py`
configuration and not part of the current C6 candidate.

## Mapping To C6

C6 contains the base official model-path features: classification objective,
velocity-free matching, transformer structure and initialization, Caffe
ResNet-101, modulated DCNv2, FPN, FCOS3D initialization, GridMask, photometric
distortion, and augmentation-free evaluation.

C6 screening and confirmation are not full official training reproductions:

- They use subset datasets rather than the complete train split.
- They use physical batch 2 on one GPU, giving effective global batch 2.
- They use no gradient accumulation.
- Their 60-epoch screening schedule differs from the official 24 epochs and
  500-iteration warmup.
- Candidate selection uses nearest-center diagnostics rather than official
  nuScenes mAP and NDS.
- C6 does not use the optional CBGS dataset wrapper.

The full-training branch adds opt-in official GT semantics, deterministic
CBGS, and gradient accumulation. Its selected physical batch 4 and two
accumulation steps give effective batch 8 on one GPU. Accumulated microbatches
still normalize their DETR losses independently before gradient averaging, so
this is a close effective-batch reproduction rather than bit-identical DDP
optimization.

A full C6 reproduction decision must therefore state whether it targets the
base official model or the higher-scoring CBGS variant, and whether it targets
the exact distributed effective batch or a practical single-GPU approximation.
