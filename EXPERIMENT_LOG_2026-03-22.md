# DETR3D Experiment Log

Updated: 2026-03-22 23:26:19 KST

## Branch
- current branch: `exp/official-repo-like`
- comparison branch: `exp/paper-text-faithful`
- purpose: official-repo-like DETR3D path for reproduction and continued debugging

## Goal
- One-sample overfit with pure PyTorch DETR3D.
- Stay close to paper + official repo.

## Best Settings
- Classification-oriented default:
  - `focal_alpha=0.5`
  - `focal_gamma=1.5`
  - `epochs=60`
  - `loss_cls_weight=1.0`
  - `scheduler=none`
- Geometry-oriented fallback:
  - `focal_alpha=0.25`
  - `focal_gamma=2.0`
  - `epochs=80`
  - `loss_cls_weight=1.0`
  - `scheduler=none`

## Paper-Faithful Rollback
- Reverted from the overfit branch:
  - learned weighted cross-attention back to valid-feature averaging
  - iterative reference refinement back to per-layer fresh reference prediction
  - dedicated initial `reference_points` layer back to layerwise `ref_branches`
- Kept:
  - cls-branch `LayerNorm`
  - improved loss normalization
  - standalone runner and diagnostics

## Paper-Faithful Baseline
- Run:
  - `focal_alpha=0.25`
  - `focal_gamma=2.0`
  - `epochs=40`
  - `scheduler=none`
- Result:
  - `loss_cls=0.229`
  - `loss_bbox=5.090`
  - mean center distance: `15.858 m`
  - median center distance: `4.513 m`
  - class matches: `3/10`
- Takeaway:
  - the paper-faithful rollback is much weaker on one-sample overfit than the overfit-optimized branch
  - this gives a clean baseline for paper-faithful follow-up experiments

## Paper-Faithful With Paper-Like Schedule
- Run:
  - `focal_alpha=0.25`
  - `focal_gamma=2.0`
  - `epochs=40`
  - `scheduler=multistep`
  - milestones `8 11`, gamma `0.1`
- Result:
  - `loss_cls=0.432`
  - `loss_bbox=5.434`
  - mean center distance: `13.791 m`
  - median center distance: `3.029 m`
  - class matches: `3/10`
- Takeaway:
  - paper-like LR decay did not materially fix the paper-faithful branch
  - the major gap is still architectural behavior, not just schedule choice

## Current Files
- [detr3d.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/detr3d.py)
- [detr3d_head.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/heads/detr3d_head.py)
- [detr3d_loss.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/losses/detr3d_loss.py)
- [cross_attention.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/transformer/cross_attention.py)
- [overfit_one_batch.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/scripts/overfit_one_batch.py)
- [detr3d_e2e_walkthrough.ipynb](/home/user/workspace/ML_study/implementations/detr3d/notebooks/detr3d_e2e_walkthrough.ipynb)

## Former Baseline
- Notebook after cleanup:
  - `num_queries=100`
  - auxiliary losses on
  - `lr=2e-4`, `backbone_lr_mult=0.1`, `weight_decay=0.01`
- Result:
  - mean center distance: `12.987 m`
  - median center distance: `1.824 m`
  - class matches: `5/10`

## Exp 1: Geometry / Refinement
- Change:
  - learned camera/level attention in cross-attention
  - reference-point positional encoding
  - iterative reference refinement instead of fresh refs every layer
- Result:
  - `loss_bbox=2.579`
  - `debug_bbox_center=1.562`
  - `matched_bbox_cost=11.433`
  - mean center distance: `6.538 m`
  - median center distance: `2.831 m`
  - class matches: `3/10`
- Takeaway:
  - huge geometry improvement
  - classification still weak

## Exp 2: Classification Weight = 2.0
- Change:
  - added explicit `loss_cls_weight`
  - matched official weight `2.0`
- Result:
  - `loss_cls=1.081`
  - `loss_bbox=2.640`
  - mean center distance: `6.818 m`
  - median center distance: `2.929 m`
  - class matches: `3/10`
  - top predictions: almost all `truck`
- Takeaway:
  - did not help
  - keep geometry changes, do not prefer this classification setting

## Tooling
- Added standalone runner:
  - [overfit_one_batch.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/scripts/overfit_one_batch.py)
- Purpose:
  - avoid rerunning full notebook
  - save JSON summary
  - print GT classes, top predictions, nearest matches

## Exp 3: Classification Weight = 1.0
- Change:
  - reran same setup with `--loss-cls-weight 1.0`
- Result:
  - `loss_cls=0.552`
  - `loss_bbox=3.905`
  - `debug_bbox_center=2.837`
  - `matched_bbox_cost=16.741`
  - mean center distance: `7.533 m`
  - median center distance: `3.638 m`
  - class matches: `3/10`
  - top predictions: almost all `traffic_cone`
- Takeaway:
  - weaker cls weight flipped collapse from `truck` to `traffic_cone`
  - geometry got worse than the previous best run
  - class matches did not improve

## New Next
- Keep current geometry changes.
- Next experiment targets classification branch parity.
- Change:
  - add LayerNorm in hidden classification MLP layers to match official DETR3D head more closely

```bash
python detr3d/scripts/overfit_one_batch.py \
  --dataroot /home/user/datasets/nuscenes \
  --version v1.0-trainval \
  --sample-index 0 \
  --image-height 832 \
  --image-width 1472 \
  --num-queries 100 \
  --epochs 40 \
  --lr 2e-4 \
  --backbone-lr-mult 0.1 \
  --weight-decay 0.01 \
  --loss-cls-weight 1.0 \
  --grad-clip-norm 35 \
  --output-json outputs/overfit_one_sample.json
```

## Exp 4: Classification MLP With LayerNorm
- Change:
  - added `LayerNorm` in hidden cls MLP layers
- Result:
  - `loss_cls=0.415`
  - `loss_bbox=3.557`
  - `debug_bbox_center=2.506`
  - `matched_bbox_cost=15.455`
  - mean center distance: `8.383 m`
  - median center distance: `2.549 m`
  - class matches: `6/10`
  - top predictions now mixed: `traffic_cone`, `truck`, `pedestrian`
- Takeaway:
  - clear classification improvement
  - some geometry regression versus best geometry run
  - still the most balanced result so far

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
- Next experiment:
  - try `--loss-cls-weight 2.0` again with the improved cls branch
- Reason:
  - previous `2.0` test was before cls-branch parity fix
  - now it may improve classes without the earlier full collapse

## Exp 5: LayerNorm Cls Branch + Classification Weight = 2.0
- Change:
  - reran with cls LayerNorm kept
  - switched `--loss-cls-weight 2.0`
- Result:
  - `loss_cls=0.833`
  - `loss_bbox=4.010`
  - `debug_bbox_center=2.952`
  - `matched_bbox_cost=17.186`
  - mean center distance: `8.089 m`
  - median center distance: `6.904 m`
  - class matches: `3/10`
  - top predictions: almost all `traffic_cone`
- Takeaway:
  - worse than LayerNorm + `loss_cls_weight=1.0`
  - do not use `2.0` with the current local cls branch

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
  - `loss_cls_weight=1.0`
- Next experiment:
  - make classification target plumbing closer to official MMDet DETR3D
- Reason:
  - class collapse remains the blocker
  - weight sweeps are not fixing it
  - next likely gap is target/no-object handling rather than pure scaling

## Exp 6: Loss Target Plumbing Closer To MMDet
- Change:
  - flattened cls loss to DETR/MMDet-style `(N*Q, C)` path
  - kept integer labels with background index
  - applied per-query label weights with the same DETR-style avg factor
- Result:
  - `loss_cls=0.413`
  - `loss_bbox=2.639`
  - `debug_bbox_center=1.605`
  - `matched_bbox_cost=11.669`
  - mean center distance: `13.802 m`
  - median center distance: `9.643 m`
  - class matches: `3/10`
  - top predictions: almost all `truck`
- Takeaway:
  - failed direction
  - training collapsed back toward single-class `truck`
  - keep the earlier cls LayerNorm run as the best balanced setup

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
  - `loss_cls_weight=1.0`
- Next experiment:
  - target classification branch behavior without changing target semantics again

## Exp 7: Dedicated Initial Reference-Point Layer
- Change:
  - reverted failed Exp 6 loss-target change
  - replaced leftover `ref_branches[0]` init path with a dedicated `reference_points` linear layer
  - updated trainer debug hook to track the new parameter
- Result:
  - `loss_cls=0.419`
  - `loss_bbox=2.414`
  - `debug_bbox_center=1.363`
  - `matched_bbox_cost=10.823`
  - mean center distance: `4.921 m`
  - median center distance: `2.720 m`
  - class matches: `6/10`
- Takeaway:
  - new best run so far
  - geometry improved clearly without losing the better cls behavior
  - dedicated initial reference-point layer is worth keeping

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
  - original focal-target semantics
  - `loss_cls_weight=1.0`
- Next experiment:
  - test whether early LR decay is limiting one-sample overfit

## Exp 8: No LR Decay In One-Sample Runner
- Change:
  - added scheduler controls to [overfit_one_batch.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/scripts/overfit_one_batch.py)
  - next run uses `--scheduler none`
- Result:
  - `loss_cls=0.187`
  - `loss_bbox=1.793`
  - `debug_bbox_center=0.765`
  - `matched_bbox_cost=8.281`
  - mean center distance: `1.958 m`
  - median center distance: `1.227 m`
  - class matches: `7/10`
- Takeaway:
  - new best run so far
  - early LR decay was limiting one-sample overfit
  - geometry is now much stronger, class confusion remains mostly `pedestrian` vs `traffic_cone` and one `car`

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
  - dedicated initial reference-point layer
  - original focal-target semantics
  - `loss_cls_weight=1.0`
  - `--scheduler none`
- Next experiment:
  - extend the one-sample run length before changing architecture again

## Exp 9: Longer One-Sample Run (80 Epochs, No LR Decay)
- Change:
  - kept the current best setup
  - increased one-sample overfit run from `40` to `80` epochs
- Result:
  - `loss_cls=0.161`
  - `loss_bbox=1.523`
  - `debug_bbox_center=0.707`
  - `matched_bbox_cost=7.212`
  - mean center distance: `3.403 m`
  - median center distance: `1.998 m`
  - class matches: `8/10`
- Takeaway:
  - classification improved further
  - one `car` and both `pedestrian` mismatches were reduced to only the two `pedestrian` mismatches
  - geometry metrics are mixed versus the 40-epoch no-decay run, but overall fit quality is better

## New Next
- Keep:
  - geometry/refinement changes
  - cls LayerNorm change
  - dedicated initial reference-point layer
  - original focal-target semantics
  - `loss_cls_weight=1.0`
  - `--scheduler none`
- Next experiment:
  - check whether the remaining misses are partly caused by `max_boxes=50` filtering in the summary

## Exp 10: Evaluate All 100 Queries In Summary
- Change:
  - keep the same 80-epoch no-decay best setup
  - rerun summary with `--max-boxes 100`
- Result:
  - `max_boxes=100` applied correctly
  - `num_pred=100`
  - mean center distance: `1.236 m`
  - median center distance: `1.150 m`
  - class matches: `8/10`
- Takeaway:
  - top-50 filtering was hiding some geometry quality, but not the remaining class errors
  - the two remaining misses are real `pedestrian -> traffic_cone` confusions

## New Next
- Keep the current best setup.
- Next experiment:
  - retest official `loss_cls_weight=2.0` under the new stronger baseline

## Exp 11: Official Classification Weight On Strong Baseline
- Change:
  - keep the current 80-epoch no-decay best setup
  - switch `--loss-cls-weight` from `1.0` to `2.0`
- Result:
  - `loss_cls=0.310`
  - `loss_bbox=1.496`
  - mean center distance: `1.394 m`
  - median center distance: `1.189 m`
  - class matches: `8/10`
- Takeaway:
  - not a real improvement
  - `pedestrian -> traffic_cone` confusion remained
  - keep `loss_cls_weight=1.0` as the preferred setting

## New Next
- Keep the current best setup with `loss_cls_weight=1.0`.
- Next experiment:
  - diagnose whether the remaining pedestrian misses are true classification failures or nearest-neighbor summary artifacts

## Exp 12: Nearest Same-Class Diagnostic
- Change:
  - extended [overfit_one_batch.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/scripts/overfit_one_batch.py) to report nearest same-class prediction for each GT
- Result:
  - `class_matches=8/10`
  - for both missed pedestrians, nearest overall prediction was a nearby `traffic_cone`
  - nearest same-class `pedestrian` prediction existed, but was far away:
    - about `11.386 m`
    - about `10.641 m`
- Takeaway:
  - this is a real classification/localization gap for pedestrians
  - not just a summary artifact or score-ordering issue

## New Next
- Keep the current best setup.
- Next experiment:
  - increase focal positive weighting without changing the geometry path

## Exp 13: Higher Focal Alpha For Hard Positives
- Change:
  - added `--focal-alpha` and `--focal-gamma` to [overfit_one_batch.py](/home/user/workspace/ML_study/implementations/detr3d/detr3d/scripts/overfit_one_batch.py)
  - next run uses `--focal-alpha 0.5`
- Result:
  - `loss_cls=0.213`
  - `loss_bbox=1.680`
  - mean center distance: `2.484 m`
  - median center distance: `1.237 m`
  - class matches: `9/10`
- Takeaway:
  - best class-match result so far
  - both pedestrians are now correct
  - one traffic_cone flipped to `pedestrian`
  - geometry regressed somewhat, especially one truck

## New Next
- Keep both strong baselines in mind:
  - `alpha=0.25`: better geometry, `8/10`
  - `alpha=0.5`: better classification, `9/10`
- Next experiment:
  - interpolate between them with `--focal-alpha 0.35`

## Exp 14: Interpolate Focal Alpha To 0.35
- Change:
  - reran the current best setup with `--focal-alpha 0.35`
- Result:
  - `loss_cls=0.171`
  - `loss_bbox=1.765`
  - mean center distance: `2.365 m`
  - median center distance: `1.524 m`
  - class matches: `8/10`
- Takeaway:
  - did not preserve the `9/10` gain
  - both pedestrians fell back to nearby `traffic_cone`
  - prefer `alpha=0.5` over `0.35` if classification is the priority

## New Next
- Current strong options:
  - `alpha=0.25`: stronger geometry
  - `alpha=0.5`: stronger classification
- Next experiment:
  - try a shorter no-decay run with `alpha=0.5` to see if it keeps `9/10` while improving geometry

## Exp 15: Alpha 0.5 With Shorter Run (60 Epochs)
- Change:
  - kept `focal_alpha=0.5`
  - shortened run from `80` to `60` epochs
- Result:
  - `loss_cls=0.229`
  - `loss_bbox=1.905`
  - mean center distance: `1.977 m`
  - median center distance: `1.452 m`
  - class matches: `9/10`
- Takeaway:
  - preserved `9/10`
  - geometry improved relative to the 80-epoch `alpha=0.5` run
  - remaining miss is now one `traffic_cone -> pedestrian`

## New Next
- Current preferred classification-oriented setup:
  - `focal_alpha=0.5`
  - `epochs=60`
- Next experiment:
  - small sweep around this new best point, starting with `epochs=50`

## Exp 16: Alpha 0.5 With 50 Epochs
- Change:
  - kept `focal_alpha=0.5`
  - shortened run from `60` to `50` epochs
- Result:
  - `loss_cls=0.247`
  - `loss_bbox=2.385`
  - mean center distance: `3.147 m`
  - median center distance: `2.434 m`
  - class matches: `8/10`
- Takeaway:
  - too short
  - lost the `9/10` gain and degraded geometry
  - keep `epochs=60` as the better classification-oriented setting

## Exp 17: Alpha 0.5 With Lower Gamma
- Change:
  - keep `focal_alpha=0.5`
  - keep `epochs=60`
  - lower `focal_gamma` from `2.0` to `1.5`
- Result:
  - `loss_cls=0.312`
  - `loss_bbox=1.663`
  - mean center distance: `1.733 m`
  - median center distance: `1.376 m`
  - class matches: `9/10`
- Takeaway:
  - preserved `9/10`
  - geometry improved versus the previous `alpha=0.5, gamma=2.0, epochs=60` run
  - remaining miss is still one `traffic_cone -> pedestrian`

## New Next
- Current best classification-oriented setup:
  - `focal_alpha=0.5`
  - `focal_gamma=1.5`
  - `epochs=60`
- Next step:
  - one final narrow attempt at `10/10`: increase to `epochs=70`

## Exp 18: Final Narrow Push To 10/10
- Change:
  - keep `focal_alpha=0.5`
  - keep `focal_gamma=1.5`
  - increase `epochs` from `60` to `70`
- Result:
  - `loss_cls=0.328`
  - `loss_bbox=1.719`
  - mean center distance: `2.525 m`
  - median center distance: `1.908 m`
  - class matches: `8/10`
- Takeaway:
  - final push regressed
  - did not reach `10/10`
  - keep `alpha=0.5, gamma=1.5, epochs=60` as the best classification-oriented setting

## Conclusion
- Geometry-oriented best:
  - `focal_alpha=0.25`
  - `focal_gamma=2.0`
  - `epochs=80`
  - `scheduler=none`
- Classification-oriented best:
  - `focal_alpha=0.5`
  - `focal_gamma=1.5`
  - `epochs=60`
  - `scheduler=none`
