# AGENTS.md

## Project Goal

This repository is a pure PyTorch DETR3D reproduction. The practical goal is to reproduce official DETR3D behavior and training quality while keeping the implementation independent from MMDetection/MMDetection3D.

When the paper text and official repo differ or the paper is underspecified, prefer the official repo/config behavior for reproduction work.

## Current Working Context

- Continue from the official-repo-like implementation path.
- Treat `COMMAND_GUIDE.md` as the canonical command reference.
- The faithful C6 model is the accepted baseline on `main`: 34.58 mAP and 42.15 NDS on the full nuScenes validation split.
- The lightweight one-sample regression check is `detr3d/scripts/overfit_one_batch.py` with the seeded deterministic command in `COMMAND_GUIDE.md`; it is not a faithful-quality metric.
- Do not use old session handoff notes as active guidance; their useful content has been consolidated here and in `COMMAND_GUIDE.md`.

## Current Baseline

Canonical one-sample result recorded for the seeded command:

- `class_matches = 7/10`
- `mean_center_distance = 3.5343 m`
- `median_center_distance = 3.4936 m`
- `loss_cls = 0.3412`
- `loss_bbox = 1.9720`

These values were refreshed on the final faithful implementation. Use them only
as a deterministic implementation regression reference. Use official nuScenes
mAP/NDS for model-quality and promotion decisions.

## Implementation Principles

- Preserve the official-style encoded `10D` internal box training path unless intentionally running a controlled experiment.
- Dataset GT boxes are semantic `9D`; training/matching/loss operate on encoded `10D`; diagnostics decode predictions back to semantic `9D` for readability.
- Prefer official-style filtered GT for current quality experiments: keep boxes inside `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` and drop zero-point annotations with `--filter-gt-by-range --filter-zero-point-gt`.
- Preserve official-like learned cross-attention weighting, iterative reference refinement, dedicated initial reference points, and classification-head normalization unless intentionally testing a targeted change.
- Make one targeted change at a time and compare against the canonical one-sample regression check.
- Avoid broad hyperparameter sweeps before checking implementation parity with the official DETR3D repo/config.

## Experiment Workflow

- Before risky model-path changes, confirm the current branch and worktree state.
- Run the canonical one-sample command from `COMMAND_GUIDE.md` before and after substantial behavioral changes when feasible.
- Before broad training, use `detr3d/scripts/benchmark_forward.py` to choose a small-training setup that uses both GPU memory and CPU data loading efficiently.
- Current quality experiments should avoid AMP. AMP improves throughput but was observed to make one-sample and small-sample training stall in this repo.
- Record only durable command/baseline updates in `COMMAND_GUIDE.md`.
- Keep historical experiment diaries out of the repo root unless they are actively needed.

## Branch And Commit Workflow

- Keep `main` as the stable faithful C6 baseline.
- Use a separate experiment branch for model, loss, matcher, data, or training changes.
- Stay on the small-training branch while changing model details, loss/matcher behavior, diagnostics, runner settings, and small-sample training commands.
- Commit small, coherent units: docs cleanup, generated-artifact cleanup, and behavioral code changes should be separate commits.
- Push the small-training branch regularly after coherent commits so experiment history is preserved remotely.
- Create a new branch only when starting a conceptually separate experiment or promoting a verified result.
- After one-sample and small-sample results are stable enough to justify scale-up, create a dedicated full-training branch from the accepted experiment commit.
- Do not mix full-training-only changes back into the small-training search branch unless they are generally useful fixes.

## Repository Hygiene

- Do not commit generated artifacts such as `__pycache__/`, `*.pyc`, checkpoints, or `outputs/`.
- Be careful in a dirty worktree. Do not revert or overwrite user changes unless explicitly requested.
- Prefer small, reviewable changes over broad rewrites.
- Keep root Markdown minimal: `README.md`, `COMMAND_GUIDE.md`, and this file should be enough for normal work.
