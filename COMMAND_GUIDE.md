# DETR3D Command Guide

This guide includes two tracks:
- the current canonical regression baseline used for reproduction on `exp/official-copy`
- older paper-oriented commands kept as secondary reference

## Branch Phase Plan

- `exp/official-copy` is the stable official-like baseline branch.
- Use a small-training/search branch, currently `exp/official-copy-experiments`, to find the best setup before broad training.
- Commit and push coherent experiment steps on the small-training branch.
- Once the best small-training setup is selected, create a full-training branch such as `exp/official-copy-full-training` from that accepted commit.
- Keep full-training changes separate from exploratory small-training changes.

## Current Canonical Baseline

For current reproduction and regression checks, use the seeded one-sample runner from the March 23 handoff note instead of the older paper-oriented defaults below.

Canonical conditions:
- baseline branch: `exp/official-copy`
- active small-training branch: `exp/official-copy-experiments`
- script: `python detr3d/scripts/overfit_one_batch.py`
- dataset: `--dataroot /home/user/datasets/nuscenes --version v1.0-trainval`
- sample: `--sample-index 0`
- image size: `--image-height 832 --image-width 1472`
- queries: `--num-queries 100`
- epochs: `--epochs 60`
- optimizer: `--lr 2e-4 --backbone-lr-mult 0.1 --weight-decay 0.01`
- loss/classification: `--loss-cls-weight 1.0 --focal-alpha 0.5 --focal-gamma 1.5`
- scheduler: `--scheduler none`
- stability: `--max-boxes 100 --grad-clip-norm 35 --seed 0 --deterministic`
- output: `--output-json outputs/overfit_one_sample.json`

Canonical command:

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

Canonical expected result from `outputs/overfit_one_sample.json`:
- `class_matches = 10/10`
- `mean_center_distance = 2.3838 m`
- `median_center_distance = 2.1390 m`
- `loss_cls = 0.3164`
- `loss_bbox = 1.6248`

## Older Paper-Oriented Reference

The sections below describe the older paper-oriented package setup and should not be treated as the primary regression baseline when they conflict with the canonical settings above.

This guide now targets a more paper-oriented DETR3D package setup:
- ResNet-101 backbone
- 4-output FPN
- 900 queries
- LiDAR-frame boxes and `lidar2img`
- sigmoid focal classification
- auxiliary decoder losses enabled by default
- AdamW with backbone LR multiplier and cosine schedule

One sample means one nuScenes timestamp with 6 camera images, not one RGB image.
The notebook now imports the package modules directly, so these commands and the notebook should stay aligned.

Important practical note:
- the rebuilt defaults are much heavier than the old notebook scaffold
- default image size is now `900x1600`
- `900` queries and a ResNet-101 backbone will use much more memory and time than the earlier debug model
- for initial sanity checks, prefer `batch-size 1`

## 1. One-Sample Overfit

### Paper-Oriented Default

This is the recommended starting point for the rebuilt project.

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 1 \
  --batch-size 1 \
  --epochs 300 \
  --num-workers 0 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/overfit_1sample \
  --save-every 50 \
  --num-eval-samples 1 \
  --eval-every 50
```

### Last-Layer-Only Ablation

Use this only if you intentionally want to disable DETR3D-style auxiliary decoder supervision.

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 1 \
  --batch-size 1 \
  --epochs 300 \
  --num-workers 0 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/overfit_1sample_no_aux \
  --save-every 50 \
  --num-eval-samples 1 \
  --eval-every 50 \
  --disable-auxiliary-losses
```

Outputs:
- `outputs/overfit_1sample/history.json`
- `outputs/overfit_1sample/last_checkpoint.pt`
- `outputs/overfit_1sample/final_checkpoint.pt`
- `outputs/overfit_1sample/checkpoint_epoch_0050.pt` etc.
- `outputs/overfit_1sample/best_eval_checkpoint.pt` if eval is enabled
- `outputs/overfit_1sample/eval/epoch_0050.json` etc.
- `outputs/overfit_1sample/eval_artifacts/epoch_0050/overlays/`
- `outputs/overfit_1sample/eval_artifacts/epoch_0050/bev/`

Sanity check:
- paper-oriented default should print decoder-layer losses
- `--disable-auxiliary-losses` should remove `d0.loss_* ...`

## 2. Resume Training

Resume from the last checkpoint for another 100 epochs.
Use a saved periodic checkpoint if the previous run stopped before writing `final_checkpoint.pt`:

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 1 \
  --batch-size 1 \
  --epochs 100 \
  --num-workers 0 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/overfit_1sample_resume \
  --resume outputs/overfit_1sample/final_checkpoint.pt \
  --save-every 50 \
  --num-eval-samples 1 \
  --eval-every 50
```

## 3. Evaluate One Sample

Run notebook-style numeric diagnostics, save overlay images, and save BEV figures:

```bash
python3 eval.py \
  --checkpoint outputs/overfit_1sample/best_eval_checkpoint.pt \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 1 \
  --sample-index 0 \
  --score-threshold 0.005 \
  --max-boxes 50 \
  --save-overlay-dir outputs/overfit_1sample/overlays \
  --save-bev-dir outputs/overfit_1sample/bev
```

Outputs:
- `outputs/overfit_1sample/best_eval_checkpoint.eval.json`
- `outputs/overfit_1sample/overlays/0000_<sample_token>_overlay.png`
- `outputs/overfit_1sample/bev/0000_<sample_token>_bev.png`

## 4. Evaluate Multiple Samples

Evaluate a small slice of the dataset and aggregate results:

```bash
python3 eval.py \
  --checkpoint outputs/overfit_1sample/best_eval_checkpoint.pt \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 8 \
  --sample-indices 0,1,2,3 \
  --score-threshold 0.005 \
  --max-boxes 50 \
  --save-overlay-dir outputs/eval_multi/overlays \
  --save-bev-dir outputs/eval_multi/bev \
  --summary-out outputs/eval_multi/summary.json
```

Or evaluate the first 4 samples automatically:

```bash
python3 eval.py \
  --checkpoint outputs/overfit_1sample/best_eval_checkpoint.pt \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 8 \
  --num-eval-samples 4 \
  --score-threshold 0.005 \
  --max-boxes 50 \
  --save-overlay-dir outputs/eval_multi/overlays \
  --save-bev-dir outputs/eval_multi/bev
```

## 5. Eight-Sample Overfit

Once one-sample overfit is good, move to 8 samples:

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 8 \
  --batch-size 1 \
  --epochs 150 \
  --num-workers 0 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/overfit_8samples \
  --save-every 25 \
  --num-eval-samples 4 \
  --eval-every 25
```

If you want the same run with auxiliary decoder losses disabled:

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 8 \
  --batch-size 1 \
  --epochs 150 \
  --num-workers 0 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/overfit_8samples_no_aux \
  --save-every 25 \
  --num-eval-samples 4 \
  --eval-every 25 \
  --disable-auxiliary-losses
```

Then inspect:

```bash
python3 eval.py \
  --checkpoint outputs/overfit_8samples/best_eval_checkpoint.pt \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 8 \
  --num-eval-samples 4 \
  --score-threshold 0.005 \
  --max-boxes 50 \
  --save-overlay-dir outputs/overfit_8samples/overlays \
  --save-bev-dir outputs/overfit_8samples/bev
```

## 6. Medium-Scale Confidence Run

Before broad training, run a medium subset to confirm that the package path stays stable beyond tiny overfit experiments:

Start with `batch-size 1` if you are unsure about memory. Move to `2` only after confirming the rebuilt detector fits comfortably.

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 64 \
  --batch-size 2 \
  --epochs 75 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --pin-memory \
  --persistent-workers \
  --use-amp \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/train_64 \
  --save-every 10 \
  --num-eval-samples 8 \
  --eval-every 10
```

Then inspect:

```bash
python3 eval.py \
  --checkpoint outputs/train_64/best_eval_checkpoint.pt \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 64 \
  --num-eval-samples 8 \
  --score-threshold 0.005 \
  --max-boxes 50 \
  --save-overlay-dir outputs/train_64/overlays \
  --save-bev-dir outputs/train_64/bev
```

If you want the same run with auxiliary losses disabled:

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 64 \
  --batch-size 2 \
  --epochs 75 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --pin-memory \
  --persistent-workers \
  --use-amp \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/train_64_no_aux \
  --save-every 10 \
  --num-eval-samples 8 \
  --eval-every 10 \
  --disable-auxiliary-losses
```

## 7. Fuller Training

After the `1` sample, `8` sample, and medium-scale subset stages look stable:

Only use this after the earlier stages look sane. The rebuilt detector is much heavier than the original debug setup, so don’t treat this as a casual first run.

```bash
python3 train.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --max-samples 128 \
  --batch-size 2 \
  --epochs 50 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --pin-memory \
  --persistent-workers \
  --use-amp \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --grad-clip-norm 35 \
  --output-dir outputs/train_128 \
  --save-every 10 \
  --num-eval-samples 8 \
  --eval-every 10
```

If your GPU memory is comfortable, try `--batch-size 4`.
If you hit OOM, fall back to `--batch-size 2`, then `1`, or reduce `--num-workers`.

## What To Look For

One-sample overfit should show:
- low mean center distance
- mostly one-to-one query usage
- `w/l/h` close to GT
- strong top scores

Eight-sample overfit should show:
- no severe query collapse
- center distances still reasonable
- less small-object confusion than the one-sample run

Medium-scale confidence runs should show:
- stable eval metrics across several checkpoints
- no sudden collapse after scaling from `8` samples
- reasonable overlays and BEV plots on multiple eval samples
- better GPU utilization than the `1`-sample overfit run

Larger training should show:
- stable eval metrics across several saved checkpoints
- `best_eval_checkpoint.pt` outperforming the final checkpoint when overfitting starts

Auxiliary-loss ablation should show:
- whether disabling auxiliary losses helps or hurts this reproduction
- whether query sharing increases or decreases
- whether center error improves or gets worse

## Recommended Next Steps

After these additions, the most useful next steps are:

1. Run `1` sample paper-oriented overfit and confirm the detector can still fit one scene.
2. Run the same `1` sample setup with `--disable-auxiliary-losses` only as an ablation.
3. Run `8` sample overfit and compare `best_eval_checkpoint.pt` against `final_checkpoint.pt`.
4. Run a `64`-sample confidence check before broad training.
5. If small-object confusion remains high, focus next on:
   - class imbalance handling
   - image augmentations closer to the original pipeline
   - stronger pretrained initialization
   - more careful small-object supervision
6. Only then scale to broader training.

## Notebook Note

The notebook now imports the package backbone, neck, transformer, head, loss, and trainer instead of redefining them.
That means:
- if you change package code, the notebook follows it after rerunning the affected cells
- the old notebook-only debug commands are no longer the authoritative reference
- this guide is now the canonical run reference for both package and notebook behavior
