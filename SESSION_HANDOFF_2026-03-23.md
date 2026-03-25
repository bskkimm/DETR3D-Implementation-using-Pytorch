## Current Status

- Working branch: `exp/official-repo-like`
- Current code commit before this handoff note: `7385137` (`feat: restore official-repo-like detr3d path`)
- This branch is the one to continue using for reproduction/performance work.
- Reference-only comparison branch: `exp/paper-text-faithful`

## Branch Meaning

- `exp/official-repo-like`
  - Keeps the stronger DETR3D path that is closer to the official repo behavior in key implementation details.
  - Includes:
    - learned cross-attention weighting in `detr3d/models/transformer/cross_attention.py`
    - iterative reference refinement in `detr3d/models/detr3d.py`
    - dedicated initial `reference_points` path and cls `LayerNorm` in `detr3d/models/heads/detr3d_head.py`
    - improved DETR-style cls normalization in `detr3d/models/losses/detr3d_loss.py`

- `exp/paper-text-faithful`
  - Keeps the rollback version closer to the literal paper wording:
    - valid-feature averaging
    - fresh reference prediction each decoder layer
    - no dedicated initial reference-point layer
  - This branch was much weaker in local one-sample overfit runs.

## Important Finding

The earlier confusion around `9/10` vs `6/10` was mainly due to run-to-run variance, not loss of the official-like model state.

To make one-sample experiments reproducible, `detr3d/scripts/overfit_one_batch.py` was updated to support:

- `--seed`
- `--deterministic`
- automatic `CUBLAS_WORKSPACE_CONFIG` setup when deterministic mode is requested
- deterministic mode uses `torch.use_deterministic_algorithms(..., warn_only=True)` because strict deterministic CUDA fails for `grid_sample` backward

## Canonical One-Sample Command

Use this exact command on `exp/official-repo-like`:

```bash
python detr3d/scripts/overfit_one_batch.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --sample-index 0 \
  --image-height 832 \
  --image-width 1472 \
  --num-queries 100 \
  --epochs 60 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --loss-cls-weight 1.0 \
  --focal-alpha 0.5 \
  --focal-gamma 1.5 \
  --scheduler none \
  --max-boxes 100 \
  --grad-clip-norm 35 \
  --seed 0 \
  --deterministic \
  --output-json outputs/overfit_one_sample.json
```

## Canonical Result

From `outputs/overfit_one_sample.json` with the command above:

- `class_matches = 10/10`
- `mean_center_distance = 2.3838 m`
- `median_center_distance = 2.1390 m`
- `loss_cls = 0.3164`
- `loss_bbox = 1.6248`

This is the current reproducible one-sample baseline.

## Paper vs Official Repo Conclusion

- The local branch that worked best is closer to the official DETR3D repo than to the literal paper wording.
- The paper-text-faithful rollback branch remained weak even after schedule changes.
- For future work aimed at performance reproduction, follow `exp/official-repo-like`, not the paper-text branch.

## Suggested Next Task

Do not continue one-sample tuning unless a regression appears.

Next practical step:

1. keep working on `exp/official-repo-like`
2. use the canonical seeded command above as the regression check
3. move to a tiny multi-sample sanity run
4. compare any remaining gaps only against the official DETR3D repo/config

## Full-Training Resume Note

If a long `train.py` run is already in progress and you need to stop it safely:

1. press `Ctrl+C` once
2. wait for the process to exit cleanly
3. resume from the latest explicit epoch checkpoint, not from `last_checkpoint.pt` unless that file exists

Example: if epoch 1 already wrote `outputs/full_train_official_like_2026-03-24/checkpoint_epoch_0001.pt`, resume with:

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --epochs 11 \
  --batch-size 1 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --image-height 832 \
  --image-width 1472 \
  --backbone resnet101 \
  --num-queries 100 \
  --grad-clip-norm 35 \
  --output-dir outputs/full_train_official_like_2026-03-24 \
  --save-every 2 \
  --eval-sample-indices 0,1,2,3 \
  --eval-every 2 \
  --eval-score-threshold 0.005 \
  --eval-max-boxes 100 \
  --resume outputs/full_train_official_like_2026-03-24/checkpoint_epoch_0001.pt
```

Important:

- in this codebase, `--epochs` means additional epochs after resume, not total target epochs
- resuming from epoch 1 with `--epochs 11` will continue through epoch 12 total
- if `history.json` / `last_checkpoint.pt` have not been written yet, the per-epoch checkpoint is the safe resume source

## Working Tree Notes

At handoff time there are older non-code artifacts in the repo that were not part of the clean code path:

- `notebooks/detr3d_e2e_walkthrough.ipynb` is modified
- `EXPERIMENT_LOG_2026-03-22.md` is untracked
- `PAPER_VS_OFFICIAL_REPO_NOTES.md` is untracked

They were intentionally not part of the core model preservation path.
