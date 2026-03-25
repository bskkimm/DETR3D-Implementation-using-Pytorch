# DETR3D Paper vs Official Repo Notes

Updated: 2026-03-22

## Purpose
- Distinguish the algorithm as described in the paper from the actual official DETR3D implementation.
- Record which differences are likely to matter for optimization and final performance.

## Bottom Line
- The paper gives a simplified, high-level algorithm.
- The official repo adds implementation details that are not described the same way in the paper.
- In our local experiments, the branch closer to the official repo was much stronger than the branch closer to the literal paper text.

## Main Differences

### 1. Feature Aggregation In The Detection Head
- Paper:
  - project reference points into each camera and FPN level
  - bilinear sample features
  - average valid sampled features across cameras and levels
- Official repo:
  - predicts learned attention weights for sampled features
  - uses extra output projection
  - uses position encoding from reference points
- Practical impact:
  - stronger learned weighting can help the model suppress bad views and emphasize informative views
  - this likely improves optimization and makes multi-view fusion more expressive
  - this is one likely reason the official-style path trains better than plain averaging

### 2. Reference-Point Flow Across Decoder Layers
- Paper:
  - each decoder layer predicts a reference point from the current query
  - reads as fresh reference prediction per layer
- Official repo:
  - initializes reference points explicitly
  - then iteratively refines them using regression outputs and inverse-sigmoid updates
- Practical impact:
  - iterative refinement gives smoother geometric updates across layers
  - this likely stabilizes localization and reduces query drift
  - in our repo, this was one of the biggest improvements to one-sample overfit

### 3. Detection Head Structure
- Paper:
  - says classification and regression sub-networks have two hidden FC layers
  - says LayerNorm is used in the detection head
- Official repo:
  - uses MMDet-style branch structure and LayerNorm in practice
  - reference-point handling is partly owned by transformer/head interaction, not just a simple per-layer `Phi_ref`
- Practical impact:
  - LayerNorm helps optimization stability
  - branch structure and head-transformer coupling matter more than the paper text suggests

### 4. Loss / Assignment Description
- Paper:
  - presents a clean set-to-set formulation
  - box loss and classification loss are described at a high level
- Official repo:
  - uses MMDet-style focal-loss plumbing
  - uses implementation-specific averaging / weighting details
  - uses practical training settings beyond the minimal paper equations
- Practical impact:
  - these details strongly affect whether training collapses or converges
  - even when the conceptual loss is the same, reduction and normalization details matter a lot

### 5. Training Recipe
- Paper:
  - reports full training setup and schedule at the experiment level
- Official repo:
  - actual config is what reproduces reported behavior
  - includes concrete optimizer, schedule, and module wiring choices
- Practical impact:
  - reproducing reported performance requires following the config and code, not only the paper prose
  - a paper-faithful but code-divergent implementation can underperform even if it sounds conceptually correct

## Why The Official Repo Likely Performs Better
- It is more optimized for trainability than the simplified paper description.
- Learned feature weighting is richer than plain valid-feature averaging.
- Iterative reference refinement is likely easier to optimize than predicting independent fresh references every layer.
- The repo includes practical engineering details that reduce instability and class-collapse behavior.
- In short:
  - the paper explains the method
  - the repo is the performance-critical implementation

## Our Local Evidence
- Official-style / overfit-optimized branch:
  - much stronger one-sample overfit
  - reached up to `9/10` class matches
- Paper-faithful rollback branch:
  - much weaker one-sample overfit
  - stayed around `3/10` class matches
- This does not prove the paper is wrong.
- It shows that the official implementation contains important optimization behavior not captured by the simplified wording alone.

## Recommendation
- If the goal is paper-text faithfulness:
  - use the paper-faithful branch as the reference
- If the goal is reproducing official DETR3D behavior or performance:
  - treat the official repo and config as the source of truth
  - do not assume the paper wording is sufficient for implementation parity

## References
- Official transformer:
  - https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/mmdet3d_plugin/models/utils/detr3d_transformer.py
- Official head:
  - https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/mmdet3d_plugin/models/dense_heads/detr3d_head.py
- Official config:
  - https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/configs/detr3d/detr3d_res101_gridmask.py
