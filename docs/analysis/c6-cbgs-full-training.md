# C6 CBGS Full Training

Status: Cancelled before epoch 1

Branch: `exp/c6-cbgs-full-training`

Launch commit: `72c419e`

Started: 2026-07-17 22:05 JST

tmux session: `detr3d-c6-full`

Output: `outputs/c6_cbgs_full_v1`

MLflow run: `0f1fbe7f134143bbaa6a8565b19720a8`

The run was stopped after selecting base C6 without CBGS to reduce expected
single-GPU runtime from approximately 12 days to approximately 3 days. No
epoch checkpoint was produced.

## Decision

Train the official-config-oriented C6 model with the optional upstream CBGS
dataset wrapper. Use physical batch 4 and two gradient-accumulation steps on
one RTX PRO 6000, giving effective batch 8 to match the paper's eight GPUs
with one sample per GPU.

This run follows the later released 24-epoch cosine config rather than the
paper's 12-epoch step schedule, consistent with the project rule to prefer
released repository behavior when the two differ.

## Fixed Inputs

- Dataset: `/home/beomseokkim2/dataset/nuscenes`, `v1.0-trainval`, full train split.
- Base train samples: 28,130.
- Deterministic CBGS samples per epoch: 123,580.
- CBGS fingerprint for seed 0: `f4b13f792b6860aa266d9f6092c9677c1fb73e3275719460e3e577b23bf546c9`.
- FCOS3D checkpoint: `checkpoints/fcos3d.pth`.
- FCOS3D SHA-256: `b6cd0590879adc21d2a54bebde44811391f006434a2ffcdc6b41480b2e95be48`.
- Model: C6 base stack plus CBGS; no AMP.

## Optimization

| Setting | Value |
|---|---:|
| Physical batch | 4 |
| Accumulation steps | 2 |
| Effective batch | 8 |
| Epochs | 24 |
| Optimizer | AdamW |
| Base LR | `2e-4` |
| Backbone LR | `2e-5` |
| Weight decay | `0.01` |
| Warmup | 500 optimizer updates |
| Warmup ratio | `1/3` |
| Schedule | Cosine to `1e-3` of initial LR |
| Gradient clipping | L2 norm 35 |
| AMP | Disabled |
| Seed | 0 |

The CBGS map has 123,584 class memberships and samples 12,358 entries per
class. Fifty selected entries that become empty after official BEV filtering
are replaced deterministically by nonempty base samples.

## Capacity Check

The exact C6 image path at batch 4 completed a real FP32 forward/backward
benchmark:

- Throughput: 3.07 samples/s.
- Peak CUDA allocated: 90.97GB.
- Peak CUDA reserved: 91.44GB.
- GPU capacity: approximately 96GB.

The configuration fits but has limited memory headroom. Do not add AMP,
in-training image artifacts, a larger batch, or additional concurrent GPU
work. At measured throughput, 24 CBGS epochs are expected to require roughly
11-13 days before thermal pauses and official validation.

## Launch Command

```bash
/home/beomseokkim2/miniconda3/envs/torch_env/bin/python train.py \
  --dataroot /home/beomseokkim2/dataset/nuscenes \
  --version v1.0-trainval \
  --dataset-split train \
  --epochs 24 \
  --batch-size 4 \
  --accumulation-steps 2 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers \
  --image-height 900 \
  --image-width 1600 \
  --official-gt-semantics \
  --official-image-backbone \
  --official-image-preprocessing \
  --disable-pretrained-backbone \
  --init-fcos3d-checkpoint checkpoints/fcos3d.pth \
  --grid-mask \
  --photometric-distortion \
  --cbgs \
  --num-queries 900 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --loss-cls-weight 2.0 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --bg-cls-weight 0.0 \
  --scheduler cosine \
  --scheduler-total-epochs 24 \
  --warmup-steps 500 \
  --warmup-ratio 0.3333333333333333 \
  --min-lr-ratio 0.001 \
  --grad-clip-norm 35 \
  --save-every 1 \
  --output-dir outputs/c6_cbgs_full_v1 \
  --seed 0 \
  --mlflow \
  --mlflow-tracking-uri sqlite:////home/beomseokkim2/kim/detr3d/mlflow.db \
  --mlflow-experiment detr3d-full-training \
  --mlflow-run-name c6_cbgs_full_v1 \
  --thermal-action pause \
  --max-gpu-temp 90 \
  --max-cpu-temp 90 \
  --resume-gpu-temp 80 \
  --resume-cpu-temp 80
```

## Resume Command

Resume only from a completed epoch checkpoint. Omit FCOS3D initialization
because the training checkpoint already contains the complete model.

```bash
/home/beomseokkim2/miniconda3/envs/torch_env/bin/python train.py \
  --dataroot /home/beomseokkim2/dataset/nuscenes \
  --version v1.0-trainval \
  --dataset-split train \
  --epochs REMAINING_EPOCHS \
  --batch-size 4 \
  --accumulation-steps 2 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --pin-memory \
  --persistent-workers \
  --image-height 900 \
  --image-width 1600 \
  --official-gt-semantics \
  --official-image-backbone \
  --official-image-preprocessing \
  --disable-pretrained-backbone \
  --grid-mask \
  --photometric-distortion \
  --cbgs \
  --num-queries 900 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --loss-cls-weight 2.0 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --bg-cls-weight 0.0 \
  --scheduler cosine \
  --scheduler-total-epochs 24 \
  --warmup-steps 500 \
  --warmup-ratio 0.3333333333333333 \
  --min-lr-ratio 0.001 \
  --grad-clip-norm 35 \
  --save-every 1 \
  --output-dir outputs/c6_cbgs_full_v1 \
  --resume outputs/c6_cbgs_full_v1/checkpoint_epoch_XXXX.pt \
  --seed 0 \
  --mlflow \
  --mlflow-tracking-uri sqlite:////home/beomseokkim2/kim/detr3d/mlflow.db \
  --mlflow-experiment detr3d-full-training \
  --mlflow-run-name c6_cbgs_full_v1_resume \
  --thermal-action pause \
  --max-gpu-temp 90 \
  --max-cpu-temp 90 \
  --resume-gpu-temp 80 \
  --resume-cpu-temp 80
```

## Official Validation

After training, export every validation token and run the nuScenes devkit:

```bash
/home/beomseokkim2/miniconda3/envs/torch_env/bin/python eval.py \
  --checkpoint outputs/c6_cbgs_full_v1/final_checkpoint.pt \
  --dataroot /home/beomseokkim2/dataset/nuscenes \
  --version v1.0-trainval \
  --dataset-split val \
  --nuscenes-results-out outputs/c6_cbgs_full_v1/official_val/results_nusc.json \
  --run-nuscenes-eval \
  --nuscenes-eval-set val \
  --nuscenes-eval-output-dir outputs/c6_cbgs_full_v1/official_val/metrics \
  --device cuda
```

The official result is the nuScenes `metrics_summary.json` mAP/NDS output, not
the nearest-center screening diagnostic.
