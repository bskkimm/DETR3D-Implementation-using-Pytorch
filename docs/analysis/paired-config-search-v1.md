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

Pending.
