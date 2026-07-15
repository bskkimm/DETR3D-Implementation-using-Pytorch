# Full-Training 24-Epoch Root-Cause Analysis

Status: Open

Date: 2026-07-15

Branch: `exp/official-copy-experiments`

Baseline commit: `52f7413`

Run: `outputs/full_train_900x1600_q900_bs3_noamp_filtered_cosine_24e_w2`

Tracking issue: https://github.com/bskkimm/DETR3D-Implementation-using-Pytorch/issues/1

## Purpose

This document tracks the diagnosis and remediation of poor qualitative and quantitative validation behavior from the first 24-epoch full nuScenes run. It is the durable technical record for this incident. GitHub issues should track implementation work and link back to this document.

## Observed Symptoms

- The generated BEV and camera overlays contain far more predicted boxes than GT boxes.
- Many predicted centers are visibly displaced from their nearest GT centers.
- Predicted classes are frequently wrong, especially for nearby or visually similar classes.
- The custom full-validation result reports 117,032 nearest-prediction class matches for 138,057 GT boxes (84.77%).
- The custom full-validation mean nearest-center distance is 1.4528 m, but the metric does not penalize false positives and permits many-to-one assignments.

## Evidence

The full-validation diagnostic used a score threshold of 0.005 and a maximum of 100 boxes:

| Measurement | Result |
|---|---:|
| Mean GT boxes per sample | 22.94 |
| Mean rendered predictions per sample | 99.41 |
| Samples with exactly 100 rendered predictions | 5,869 / 6,019 |
| Predictions rendered across validation | 598,341 |

Post-hoc counts among the retained top-100 predictions:

| Score threshold | Mean predictions per sample |
|---:|---:|
| 0.005 | 99.41 |
| 0.10 | 48.99 |
| 0.30 | 27.31 |
| 0.50 | 19.01 |

Rerendering validation sample 0 at a 0.5 threshold reduced the image from 100 predictions to 23 predictions for 35 GT boxes. This removed most visual clutter but retained genuine missed detections and center errors.

## Four Transformer Architecture Differences

### 1. Decoder Operation Order

Current implementation:

```text
cross-attention -> norm -> self-attention -> norm -> FFN -> norm
```

Official DETR3D:

```text
self-attention -> norm -> cross-attention -> norm -> FFN -> norm
```

References:

- Current: `detr3d/models/transformer/decoder_layer.py:51-70`
- Official config: `operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')`

Why the difference matters:

Self-attention lets object queries exchange information and compete before sampling image features. In official DETR3D, cross-attention therefore receives queries that already contain object-to-object context. The current order makes each query sample multiview features before this interaction. That can increase duplicate hypotheses, weaken query specialization, and make the sampled feature less appropriate for the eventual object assignment. Both class prediction and reference-point refinement depend on the resulting query state, so the effect can reach classification and center localization.

Confidence: High. This is a confirmed structural mismatch and a leading quality-risk candidate.

### 2. Cross-Attention Residual and Dropout Semantics

Official cross-attention returns approximately:

```text
dropout(projected_visual_feature) + input_query + position_feature
```

Current cross-attention returns:

```text
dropout(projected_visual_feature) + position_feature
```

The decoder then applies:

```text
input_query + dropout(cross_attention_result)
```

References:

- `detr3d/models/transformer/cross_attention.py:69-75`
- `detr3d/models/transformer/decoder_layer.py:54-61`

Why the difference matters:

The projected image feature receives dropout inside cross-attention and again in the decoder. With dropout 0.1, its expected retention path is approximately 0.9 x 0.9 rather than one 0.9 application. The position feature is also dropped in the current implementation, while official DETR3D adds it outside the cross-attention dropout. This changes feature scale, injects noise into the positional signal, and changes the residual boundary used by normalization. The query can still train, but it is not optimizing the same function as official DETR3D.

Confidence: High for the behavioral difference; medium-high for its contribution to validation quality.

### 3. Transformer and Cross-Attention Initialization

Official DETR3D:

- Xavier-initializes transformer parameters with more than one dimension.
- Initializes cross-attention weight projection weights and biases to zero.
- Xavier-initializes the cross-attention output projection.
- Xavier-initializes the reference-point projection with zero bias.

Current implementation only explicitly initializes classification bias, reference-point bias, and final regression bias. Other transformer layers use generic PyTorch defaults.

References:

- `detr3d/models/transformer/cross_attention.py:27-35`
- `detr3d/models/transformer/decoder.py:25-38`
- `detr3d/models/heads/detr3d_head.py:61-73`

Why the difference matters:

Zero-initialized attention logits make every camera and feature level start with equal sigmoid weight before visibility masking. Random attention logits make queries prefer arbitrary cameras and levels before learning any geometry. DETR3D training is sensitive to initialization because reference points determine where image features are sampled; poor early sampling produces weak features, which then produce weak center refinement. This can slow convergence and increase run-to-run instability.

Confidence: High. The initialization mismatch is confirmed and official code explicitly calls transformer initialization important.

### 4. Feed-Forward Network Width

Current decoder FFN:

```text
256 -> 1024 -> 256
```

Official DETR3D decoder FFN:

```text
256 -> 512 -> 256
```

Reference: `detr3d/models/transformer/decoder_layer.py:32-37`

Why the difference matters:

The current FFN has approximately twice the FFN parameters and computation per decoder layer, adding roughly 1.6 million parameters across six layers. This changes capacity, regularization, activation statistics, and optimization behavior. It also prevents exact compatibility with official transformer weights. A wider FFN is not inherently worse, so this is lower confidence as an independent root cause, but it is an unnecessary difference in a reproduction project.

Confidence: High for the mismatch; low-medium for its independent quality impact.

## Other Root Causes

### P0: Visualization and Diagnostic Decoding

`detr3d/engine/diagnostics.py` uses a 0.005 threshold, selects up to 100 max-class queries, and renders every selected prediction. It does not use official flattened query-class top-k decoding or post-center-range filtering. Its fallback also returns top-k predictions when no prediction passes the requested threshold.

Impact: This is the direct cause of the apparent 100-box explosion in generated images. It does not explain genuine high-confidence misses or center errors.

### P0: Weak Classification and Background Supervision

Full-run settings:

```text
loss_cls_weight=1.0, alpha=0.5, gamma=1.5, bg_cls_weight=0.1
```

Official settings:

```text
loss_cls_weight=2.0, alpha=0.25, gamma=2.0, bg_cls_weight=0
```

With about 23 positives and 900 queries, the local classification denominator is approximately 110.7 instead of approximately 23. This makes positive classification gradients roughly 4.8 times weaker before focal-state effects; background gradients can be weakened further by the loss-weight and alpha differences.

Impact: Poor background suppression, weak score calibration, excess nontrivial query scores, and class confusion. The final low classification loss is not evidence that classification converged well because its scale changed.

### P1: Velocity Included in Hungarian Matching

The current matcher computes L1 assignment cost over all encoded 10 dimensions. Official DETR3D uses only the first eight and excludes velocity. Local velocities are reconstructed from annotation finite differences and are included in matching at full weight even though the final bbox loss weights velocity by 0.2.

Impact: Noisy velocity can change query-to-GT assignments and corrupt both class and center supervision.

### P1: Weaker Feature Initialization and Generalization

Official DETR3D loads an FCOS3D checkpoint and uses Caffe-style ResNet-101, modulated DCNv2, GridMask, and multiview photometric augmentation. The current model uses ImageNet-only initialization, an unmodulated deformable convolution, and no equivalent augmentation.

Impact: Weaker detector features and poorer validation generalization, especially for small, distant, and rare objects.

## Ruled-Out Causes for This Run

- The run used native 900x1600 images, so the known non-native resize/projection bug was inactive.
- Official train and validation scene splits were explicitly selected.
- GT range and zero-point filters were enabled.
- GT boxes and predictions use a consistent LiDAR frame.
- AMP was disabled.
- AdamW, 500-step warmup, cosine scheduling, gradient clipping, and the backbone LR multiplier were configured as intended.

## Evaluation Limitations

The current full-validation diagnostic is not nuScenes evaluation:

- Every GT independently selects its nearest prediction.
- A prediction can be reused by multiple GT boxes.
- False positives are not penalized.
- No distance gate is applied to class matches.
- Center distance uses XYZ rather than nuScenes ground-plane distance.
- mAP, NDS, mATE, mASE, mAOE, mAVE, and mAAE are not computed.

The current checkpoint must not be described as an official nuScenes-quality result.

## Recommended Remediation Plan

| ID | Priority | Work item | Verification | Status |
|---|---|---|---|---|
| EVAL-1 | P0 | Add official-style flattened query-class decoding, post-center filtering, and nuScenes result export/evaluation | Produce per-class AP, mAP, NDS, and TP errors on val | Open |
| EVAL-2 | P0 | Separate visualization threshold from regression diagnostics and remove the threshold fallback | Images contain only boxes at or above the requested threshold | Open |
| LOSS-1 | P0 | Restore official focal settings, classification weight, and background normalization | One-sample regression passes; query score distribution shows stronger background suppression | Open |
| MATCH-1 | P0 | Exclude velocity dimensions from Hungarian assignment cost | Unit test confirms assignment uses encoded dimensions 0:8 | Open |
| ARCH-1 | P0 | Restore self-attention-before-cross-attention order | Unit test records the official operation sequence | Open |
| ARCH-2 | P0 | Restore official cross-attention residual/dropout semantics | Unit test checks one residual and one visual-feature dropout path | Open |
| ARCH-3 | P0 | Restore official Xavier and zero attention initialization | Parameter initialization tests pass | Open |
| ARCH-4 | P1 | Change FFN width from 1024 to 512 | Model structure test confirms official dimensions | Open |
| PRETRAIN-1 | P1 | Add compatible detector-level initialization or quantify the ImageNet-only gap | Controlled small-training comparison | Open |
| AUG-1 | P1 | Add GridMask and multiview photometric distortion | Controlled small-training comparison | Open |

## Experiment Gates

Every behavioral change should pass these gates before another full run:

1. Unit tests for the exact parity behavior being changed.
2. Canonical one-sample regression without material regression from the recorded baseline.
3. Fixed small-training comparison against the current checkpoint/config.
4. Official nuScenes evaluation on a fixed validation subset.
5. Full validation only after the preceding gates improve or preserve the target metrics.

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-07-15 | Do not promote the 24-epoch checkpoint as a stable baseline | Qualitative failures, non-official evaluation, and confirmed architecture/loss parity gaps |
| 2026-07-15 | Preserve the checkpoint and reports as a diagnostic baseline | They are useful for measuring whether corrections improve score distribution and localization |
| 2026-07-15 | Require official nuScenes metrics before another quality claim | Existing nearest-prediction diagnostics ignore false positives and many-to-one matching |

## Updating This Document

- Update the remediation table status when an issue is started or completed.
- Add measured results to the decision log only after verification.
- Keep commands in `COMMAND_GUIDE.md`; keep diagnosis and outcomes here.
- Link implementation commits and GitHub issues from the relevant remediation row.
