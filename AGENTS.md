# AGENTS.md

## Project Goal

This repository is a pure PyTorch DETR3D reproduction. The practical goal is to reproduce official DETR3D behavior and training quality while keeping the implementation independent from MMDetection/MMDetection3D.

When the paper text and official repo differ or the paper is underspecified, prefer the official repo/config behavior for reproduction work.

## Current Working Context

- Continue from the official-repo-like implementation path.
- Treat `COMMAND_GUIDE.md` as the canonical command reference.
- The current one-sample regression check is `detr3d/scripts/overfit_one_batch.py` with the seeded deterministic command in `COMMAND_GUIDE.md`.
- Do not use old session handoff notes as active guidance; their useful content has been consolidated here and in `COMMAND_GUIDE.md`.

## Current Baseline

Canonical one-sample result recorded for the seeded command:

- `class_matches = 10/10`
- `mean_center_distance = 2.3838 m`
- `median_center_distance = 2.1390 m`
- `loss_cls = 0.3164`
- `loss_bbox = 1.6248`

Use this as a regression reference when changing model, matcher, loss, data, or training code.

## Implementation Principles

- Preserve the official-style encoded `10D` internal box training path unless intentionally running a controlled experiment.
- Dataset GT boxes are semantic `9D`; training/matching/loss operate on encoded `10D`; diagnostics decode predictions back to semantic `9D` for readability.
- Preserve official-like learned cross-attention weighting, iterative reference refinement, dedicated initial reference points, and classification-head normalization unless intentionally testing a targeted change.
- Make one targeted change at a time and compare against the canonical one-sample regression check.
- Avoid broad hyperparameter sweeps before checking implementation parity with the official DETR3D repo/config.

## Experiment Workflow

- Before risky model-path changes, confirm the current branch and worktree state.
- Run the canonical one-sample command from `COMMAND_GUIDE.md` before and after substantial behavioral changes when feasible.
- Before broad training, use `detr3d/scripts/benchmark_forward.py` to choose a small-training setup that uses both GPU memory and CPU data loading efficiently.
- Record only durable command/baseline updates in `COMMAND_GUIDE.md`.
- Keep historical experiment diaries out of the repo root unless they are actively needed.

## Branch And Commit Workflow

- Keep `exp/official-copy` as the stable official-like baseline branch unless a better setup is intentionally promoted.
- Use a separate small-training/search branch while selecting the best setup before full training, such as `exp/official-copy-small-training` or the current `exp/official-copy-experiments` branch.
- Stay on the small-training branch while changing model details, loss/matcher behavior, diagnostics, runner settings, and small-sample training commands.
- Commit small, coherent units: docs cleanup, generated-artifact cleanup, and behavioral code changes should be separate commits.
- Push the small-training branch regularly after coherent commits so experiment history is preserved remotely.
- Create a new branch only when starting a conceptually separate experiment or when promoting a stable result to the next phase.
- After one-sample and small-sample results are stable enough to justify scale-up, create a full-training branch such as `exp/official-copy-full-training` from the accepted small-training commit.
- Do not mix full-training-only changes back into the small-training search branch unless they are generally useful fixes.

## Repository Hygiene

- Do not commit generated artifacts such as `__pycache__/`, `*.pyc`, checkpoints, or `outputs/`.
- Be careful in a dirty worktree. Do not revert or overwrite user changes unless explicitly requested.
- Prefer small, reviewable changes over broad rewrites.
- Keep root Markdown minimal: `README.md`, `COMMAND_GUIDE.md`, and this file should be enough for normal work.
