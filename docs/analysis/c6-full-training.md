# C6 Full Training

Status: Running

Branch: `exp/c6-full-training`

Launch commit: `fa22a56`

Started: 2026-07-17 23:24 JST

tmux session: `detr3d-c6-full`

MLflow run: `da5c0f1ff86b4705aa600c2c3794c0ba`

Output: `outputs/c6_full_v1`

## Decision

Train base C6 without CBGS. Use physical batch 4 and two accumulation steps on
one RTX PRO 6000, giving effective batch 8 to match the paper's eight GPUs with
one sample per GPU. Follow the later released 24-epoch cosine configuration.

## Fixed Setup

- Dataset: full 28,130-sample nuScenes train split.
- Periodic validation: fixed first 512 official validation samples every epoch.
- MLflow images: fixed validation samples 0-3 only, every epoch.
- Final evaluation: all 6,019 official validation samples using nuScenes mAP/NDS.
- FCOS3D checkpoint: `checkpoints/fcos3d.pth`.
- FCOS3D SHA-256: `b6cd0590879adc21d2a54bebde44811391f006434a2ffcdc6b41480b2e95be48`.
- Model: C6, no CBGS, no AMP.

## Optimization

| Setting | Value |
|---|---:|
| Physical batch | 4 |
| Accumulation steps | 2 |
| Effective batch | 8 |
| Optimizer updates/epoch | 3,517 |
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

The C6 batch-4 FP32 benchmark sustained 3.07 samples/s with 91.44GB peak CUDA
reserved. Expected training duration is approximately 2.5-3 days plus periodic
validation and thermal pauses.

## Launch Command

```bash
/home/beomseokkim2/miniconda3/envs/torch_env/bin/python train.py \
  --dataroot /home/beomseokkim2/dataset/nuscenes \
  --version v1.0-trainval \
  --dataset-split train \
  --val-split val \
  --max-val-samples 512 \
  --num-eval-samples 512 \
  --num-eval-artifact-samples 4 \
  --eval-every 1 \
  --eval-score-threshold 0.005 \
  --eval-max-boxes 100 \
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
  --output-dir outputs/c6_full_v1 \
  --seed 0 \
  --mlflow \
  --mlflow-tracking-uri sqlite:////home/beomseokkim2/kim/detr3d/mlflow.db \
  --mlflow-experiment detr3d-full-training \
  --mlflow-run-name c6_full_v1 \
  --thermal-action pause \
  --max-gpu-temp 90 \
  --max-cpu-temp 90 \
  --resume-gpu-temp 80 \
  --resume-cpu-temp 80
```

## Recovery

Resume only from a completed epoch checkpoint. Keep every model, data,
accumulation, scheduler, validation, MLflow, and thermal argument unchanged.
Set `--epochs` to the remaining epoch count, add `--resume` pointing to the
latest `checkpoint_epoch_XXXX.pt`, and omit `--init-fcos3d-checkpoint`.

## Official Final Evaluation

```bash
/home/beomseokkim2/miniconda3/envs/torch_env/bin/python eval.py \
  --checkpoint outputs/c6_full_v1/final_checkpoint.pt \
  --dataroot /home/beomseokkim2/dataset/nuscenes \
  --version v1.0-trainval \
  --dataset-split val \
  --nuscenes-results-out outputs/c6_full_v1/official_val/results_nusc.json \
  --run-nuscenes-eval \
  --nuscenes-eval-set val \
  --nuscenes-eval-output-dir outputs/c6_full_v1/official_val/metrics \
  --device cuda
```
