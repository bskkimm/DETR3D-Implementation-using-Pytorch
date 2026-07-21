# Paired Configuration Search V1

Status: Running

Branch: `exp/paired-config-search`

Manifest: `detr3d/experiments/paired_config_search_v1.json`

## Objective

Find a configuration that consistently improves localization and
classification over B0 before permitting another full-data training run.
Small experiments do not use MLflow. A confirmed full run will use MLflow and
thermal protection with 90 C as the critical temperature.

## Candidate Lineage

| Variant | Commit | Change |
|---|---|---|
| B0 | `c360062` | Previous training configuration plus deterministic experiment support |
| C1 | `4c396c5` | Official classification objective |
| C2 | `6e1b63c` | Velocity-free Hungarian matching |
| C3 | `361d328` | Full official transformer parity stack |
| C4 | `23cae53` | Official-compatible image architecture and preprocessing |
| C5 | `77b6469` | Official FCOS3D backbone/FPN initialization |
| C6 | `e5d263b` | GridMask and multiview photometric distortion with augmentation-free evaluation |

## Protocol

- Seeds: 0, 1, 2.
- Fixed first 256 official train samples and first 256 official validation samples.
- Token hashes are frozen in the manifest.
- 900x1600 images, 900 queries, batch size 2, no AMP.
- 60 epochs, cosine schedule, 17 warmup steps.
- Evaluation at epochs 20, 40, and 60.
- Every candidate is paired against B0 by seed and checkpoint.

## Results

### Three-Seed One-Sample Gate

All variants completed 60 epochs at native resolution without non-finite
losses or execution failures. Values are means across seeds 0, 1, and 2.

| Variant | Mean center | Mean median center | Mean class match |
|---|---:|---:|---:|
| B0 | 1.9161 m | 1.1459 m | 80.00% |
| C1 | 2.4432 m | 1.7515 m | 80.00% |
| C2 | 2.1286 m | 1.6153 m | 76.67% |
| C3 | 2.3077 m | 1.1368 m | 83.33% |
| C4 | 3.3753 m | 2.2110 m | 43.33% |
| C5 | 2.2706 m | 1.3478 m | 66.67% |
| C6 | 3.1882 m | 1.7015 m | 70.00% |

C4 shows a material one-sample regression, while C5 demonstrates that the
FCOS3D initialization recovers much of the uninitialized official image-path
gap. One-sample performance is not the promotion metric, and every variant
completed the execution gate, so all B0-C6 variants proceed to the paired
256/256 experiment as specified. Small experiments do not use MLflow.

### Three-Seed 256/256 Search

All 21 runs completed successfully. Values are means across seeds at each
checkpoint.

| Variant | Epoch | Mean center | Mean median center | Class match |
|---|---:|---:|---:|---:|
| B0 | 20 | 4.1909 m | 2.5651 m | 66.42% |
| B0 | 40 | 3.8452 m | 2.3960 m | 72.44% |
| B0 | 60 | 3.7869 m | 2.4165 m | 70.32% |
| C1 | 20 | 3.5855 m | 2.0661 m | 71.15% |
| C1 | 40 | 3.5507 m | 2.3419 m | 66.67% |
| C1 | 60 | 3.7557 m | 2.5874 m | 63.08% |
| C2 | 20 | 3.6249 m | 2.2300 m | 71.16% |
| C2 | 40 | 3.6345 m | 2.3963 m | 69.54% |
| C2 | 60 | 4.1498 m | 2.9326 m | 66.94% |
| C3 | 20 | 3.9529 m | 2.3898 m | 69.79% |
| C3 | 40 | 3.5596 m | 2.0771 m | 72.22% |
| C3 | 60 | 3.6468 m | 2.2115 m | 72.20% |
| C4 | 20 | 6.6944 m | 5.2205 m | 13.11% |
| C4 | 40 | 6.8443 m | 5.4035 m | 7.83% |
| C4 | 60 | 6.7411 m | 5.3057 m | 7.18% |
| C5 | 20 | 3.2168 m | 1.8507 m | 74.02% |
| C5 | 40 | 3.0892 m | 1.7278 m | 76.56% |
| C5 | 60 | 3.2860 m | 1.7825 m | 76.81% |
| C6 | 20 | 3.8990 m | 2.3298 m | 74.78% |
| C6 | 40 | 3.0439 m | 1.7108 m | 76.57% |
| C6 | 60 | 2.9616 m | 1.6283 m | 75.43% |

For gate application, a checkpoint win requires the paired mean to clear both
the 3% center-improvement and 3 percentage-point class-improvement thresholds,
with at least two seeds individually clearing both thresholds. "Multiple"
checkpoint wins means at least two of epochs 20, 40, and 60. This strict
interpretation does not change the promoted set compared with using positive
per-seed wins.

| Variant | Strict checkpoint wins | Multiple checkpoints | Decision |
|---|---:|---:|---|
| C1 | 1 | No | Reject |
| C2 | 0 | No | Reject |
| C3 | 1 | No | Reject |
| C4 | 0 | No | Reject |
| C5 | 3 | Yes | Promote |
| C6 | 3 | Yes | Promote |

C5 clears the gate at every checkpoint, with epoch-60 paired improvements of
13.05% in mean center distance and 6.49 percentage points in class match. C6
also clears every checkpoint and reaches epoch-60 improvements of 21.44% and
5.12 percentage points. C5 and C6 are therefore the only confirmation
candidates.

## 1024/512 Confirmation

The confirmation manifest is
`detr3d/experiments/paired_config_confirmation_v1.json`. It compares B0, C5,
and C6 with the same three seeds, training recipe, and evaluation checkpoints
as the 256/256 search. The fixed official-split prefixes are disjoint by sample
and scene.

- Train samples: 1024.
- Validation samples: 512.
- Train token hash: `212ebc7c0348b13d40af4ca9adb33a912c7155e5a65d2d4a93899c9a8aad0c71`.
- Validation token hash: `14cf2bb502682c7f3a0ac2269605aca5a06d83e7456cf7f563f9b546e2388862`.
- Candidates: C5 and C6, paired against newly trained B0 controls.
- Selection rule: the same strict paired gate used above.
