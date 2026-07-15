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
