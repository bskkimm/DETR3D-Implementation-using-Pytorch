# SESSION HANDOFF - DETR3D Debug State (Updated 2026-03-22)

## User Goal
Reproduce DETR3D as faithfully as practical in pure PyTorch, using the package as the main implementation and the notebook as the debug / one-sample overfit path.

The user still wants:
- paper-faithful architecture where possible
- pure PyTorch implementation, not a permanent MMDetection dependency
- a package and notebook that agree with each other
- an implementation that can at least overfit a tiny sample before scaling up

## Environment Context
- Repo: `/home/user/workspace/ML_study/implementations/detr3d`
- Main notebook: `/home/user/workspace/ML_study/implementations/detr3d/notebooks/detr3d_e2e_walkthrough.ipynb`
- User runs notebook in conda env `detr3d` with GPU available there.
- Codex sandbox is still broken for local shell commands with:
  - `bwrap: Unknown option --argv0`
- In this session, file reads/edits/verifications had to be done with escalated shell commands.

## Important Correction Relative To Older Handoff
The previous handoff in this file is stale in one critical way:
- it describes the later paper-literal raw `9D` training path as the current package state
- that is no longer true

Current package state at end of this session:
- GT boxes are still semantic `9D` in the dataset
- but training now uses an official-style encoded `10D` target internally again
- the head predicts encoded `10D`
- matcher and bbox loss operate in encoded `10D`
- diagnostics and trainer debug stats decode predictions back to semantic `9D` for readability

This change was made because the raw `9D` training path was empirically worse.

## What Was Established This Session

### 1. Official Repo Clarifies More Than The Paper
The user asked to inspect the official repo and compare it against the paper.

Main conclusions:
- the paper is under-specified on several optimization details
- the official repo resolves many of those ambiguities
- the biggest practical difference is box training parameterization

Official repo facts confirmed:
- official repo uses `num_query=900`
- classification loss is focal with:
  - `alpha=0.25`
  - `gamma=2.0`
  - `loss_weight=2.0`
- matcher uses:
  - `cls_cost = FocalLossCost(weight=2.0)`
  - `reg_cost = BBox3DL1Cost(weight=0.25)`
  - `iou_cost = IoUCost(weight=0.0)` as a compatibility term
- bbox loss weight is `0.25`
- optimizer uses gradient clipping `max_norm=35`
- official box training target is encoded `10D`, not raw semantic `9D`

Reference config:
- https://raw.githubusercontent.com/WangYueFt/detr3d/main/projects/configs/detr3d/detr3d_res101_gridmask.py

### 2. Current Package Was Restored Toward Official-Style Encoded 10D
Core code was changed so the package now matches the official training parameterization more closely.

Current local implementation summary:
- dataset GT: semantic `9D`
  - `[x, y, z, w, l, h, yaw, vx, vy]`
- internal training target: encoded `10D`
  - `[x, y, log(w), log(l), z, log(h), sin(yaw), cos(yaw), vx, vy]`
- head output: encoded `10D`
- matcher cost:
  - focal class cost
  - encoded `10D` L1 box cost
- loss:
  - focal classification loss
  - encoded `10D` L1 bbox loss with code weights
- diagnostics:
  - decode back to semantic `9D` for human-readable summaries

### 3. Current Local Default Loss / Matcher Settings
These are now mostly aligned with the official repo defaults.

From local files:
- `Detr3DLoss(...)` defaults:
  - `matcher_cls_weight = 2.0`
  - `matcher_bbox_weight = 0.25`
  - `loss_bbox_weight = 0.25`
  - `alpha = 0.25`
  - `gamma = 2.0`
  - `code_weights = (1,1,1,1,1,1,1,1,0.2,0.2)`
- `HungarianMatcher3D(...)` defaults:
  - `cls_weight = 2.0`
  - `bbox_weight = 0.25`
  - `alpha = 0.25`
  - `gamma = 2.0`

So the current package is not wildly off anymore. It is approximately aligned with the official repo on the main loss/matcher hyperparameters.

## Code Changes Made In This Session

### 1. Head Restored To Encoded 10D Output
File:
- `/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/heads/detr3d_head.py`

Current behavior:
- `box_dim = 10`
- head returns encoded predictions
- center refinement still uses reference-point residual logic
- output contract is encoded `10D`, not raw semantic `9D`

### 2. Loss Utilities Restored To Encode / Decode Flow
File:
- `/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/losses/loss_utils.py`

Current behavior:
- semantic `9D` GT is encoded to official-style `10D`
- encoded `10D` predictions can be decoded back to semantic `9D`
- helper still handles wrapped yaw logic for debug reporting

### 3. Matcher Restored To Encoded 10D Matching
File:
- `/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/losses/matcher.py`

Current behavior:
- uses focal-style class cost
- uses encoded `10D` L1 box cost
- matching weights default to official-style `2.0` and `0.25`
- debug matcher summaries are recorded in `last_debug`

### 4. Loss Restored To Encoded 10D Supervision
File:
- `/home/user/workspace/ML_study/implementations/detr3d/detr3d/models/losses/detr3d_loss.py`

Current behavior:
- bbox loss is computed in encoded `10D` space
- classification loss is focal with local defaults `alpha=0.25`, `gamma=2.0`
- debug metrics decode to semantic `9D` for readable center/size/yaw/velocity breakdowns

### 5. Trainer Debug Stats Decode Encoded Predictions Before Reporting
File:
- `/home/user/workspace/ML_study/implementations/detr3d/detr3d/engine/trainer.py`

Current behavior:
- prediction debug stats are reported in semantic `9D` after decode
- so printed metrics like `debug_pred_center_mean_abs` remain human-readable

### 6. Notebook Was Updated To Match The Package
File:
- `/home/user/workspace/ML_study/implementations/detr3d/notebooks/detr3d_e2e_walkthrough.ipynb`

Notebook changes already made:
- explanatory markdown now distinguishes:
  - what the paper explicitly specifies
  - what the paper leaves ambiguous
  - what the official repo specifies
  - what the local package currently implements
- one-sample training cell was aligned with the restored encoded-10D package path

## Verification Completed This Session

### Syntax / Static Verification
These files passed `python3 -m py_compile`:
- `detr3d/models/heads/detr3d_head.py`
- `detr3d/models/losses/loss_utils.py`
- `detr3d/models/losses/matcher.py`
- `detr3d/models/losses/detr3d_loss.py`
- `detr3d/engine/trainer.py`

### Runtime Smoke Verification
A forward/loss/decode smoke test on CUDA passed with the restored encoded-10D path.

Observed results during validation:
- `cls_scores_shape = (6, 1, 100, 10)`
- `bbox_preds_shape = (6, 1, 100, 10)`
- decoded predictions return semantic `9D`
- losses were finite

A one-epoch one-sample training sanity check also passed:
- backward/update succeeded
- `debug_delta_*` values were nonzero
- gradients and losses were finite

Conclusion:
- the restored encoded-10D path is numerically trainable
- current problem is optimization quality / training behavior, not basic runtime breakage

## Most Important Empirical Results From This Session

### Baseline A: Encoded 10D With `num_queries=100`
This was the best recent one-sample run.

Key outcome:
- materially better than the raw `9D` training path
- not good enough yet, but clearly improved

Main metrics from that run:
- final `loss_bbox` about `3.54`
- final `debug_bbox_center` about `2.57`
- nearest-center summary:
  - mean `8.440 m`
  - median `3.109 m`
- class matches among nearest predictions: `5/10`

Observed behavior:
- predictions were no longer all one class
- some cars were correctly matched as cars
- trucks were still poorly sized and often far away
- multiple GTs still reused a small number of queries

Interpretation:
- encoded `10D` helped significantly
- class collapse weakened but did not disappear
- query / assignment collapse still exists

### Baseline B: Encoded 10D With `num_queries=300`
This was tested because the official repo uses many more queries (`900`) and the `100`-query result still showed GT reuse onto a few active queries.

Result:
- worse than `100`
- collapse returned toward `traffic_cone`

Main metrics from that run:
- final `loss_bbox` about `5.15`
- final `debug_bbox_center` about `4.09`
- nearest-center summary:
  - mean `13.199 m`
  - median `1.291 m`
- class matches among nearest predictions: `3/10`

Observed behavior:
- more distinct queries were used
- but they all clustered around easy small-object local solutions
- trucks and cars at range got much worse

Interpretation:
- the hypothesis “too few queries is the main blocker” was not supported
- simply increasing query count is not the right next step
- `100` queries is the better current debug baseline than `300`

## Current Best Baseline To Resume From Tomorrow
Use this exact baseline as the resume point:
- encoded `10D` package path
- `num_queries = 100`
- `use_amp = False`
- current official-style default loss/matcher weights:
  - `alpha = 0.25`
  - `gamma = 2.0`
  - matcher class weight `2.0`
  - matcher bbox weight `0.25`
  - bbox loss weight `0.25`
- notebook one-sample overfit path only

Do not continue from the current `300`-query notebook state. That was an experiment and it was worse.

## What The Paper Specifies vs What It Does Not

### Paper Explicitly Specifies
- DETR-style set-to-set prediction with Hungarian matching
- focal loss for classification
- semantic 3D box concept (`R^9` style description)
- one query predicts one object

### Paper Does Not Adequately Specify
- raw semantic `9D` vs encoded training target
- exact matcher class-vs-box weights
- exact focal parameters in matcher vs loss
- exact classification normalization / averaging behavior
- exact no-object handling details
- exact implementation details needed to avoid collapse

## Where Current Local Code Aligns With Official Repo
The current package is approximately aligned with the official repo on:
- encoded `10D` training target
- focal classification loss hyperparameters
- matcher class/bbox weights
- bbox loss weight
- gradient clipping in the notebook training cell

## Where Current Local Code Still May Differ In Important Ways
Even though the broad recipe is now close, the implementation is still not guaranteed identical to MMDetection / official DETR3D behavior.

Likely remaining gaps:
1. classification-loss normalization / averaging may still differ from MMDet internals
2. matcher focal-cost implementation may be close but not bit-identical
3. head prediction / decode / iterative refinement details may still differ subtly
4. decoder auxiliary supervision behavior may differ from official training usage
5. training schedule and initialization recipe still differ from official repo
6. official repo uses `900` queries, while the best current local baseline is `100`

## Most Defensible Current Diagnosis
The pure PyTorch route is still feasible in principle, but the project is now in the phase where:
- high-level architecture is mostly close enough
- remaining failures are likely caused by exact training plumbing differences, not obvious missing components

The strongest remaining suspects are:
1. classification normalization / averaging mismatch relative to official MMDet path
2. still-not-identical matcher / loss plumbing
3. subtle head / refinement behavior mismatch
4. training recipe differences beyond the main hyperparameters

## Recommended Next Steps For The Next Session

### Immediate First Step
Revert the notebook one-sample training cell from `num_queries=300` back to `num_queries=100`.

Reason:
- `300` queries was worse than `100`
- `100` is the best current reproducible baseline

### Then Do This In Order
1. Freeze the `100`-query encoded-10D baseline.
2. Compare our classification-loss normalization against the official `Detr3DHead` / MMDet path line by line.
3. Compare our matcher focal-cost implementation against the official repo line by line.
4. Compare our head output ordering and iterative refinement semantics against the official repo line by line.
5. Check whether auxiliary decoder-layer supervision behavior matches the intended official behavior.
6. Make only one targeted change at a time.
7. Re-run the one-sample overfit probe after each change.

### Most Important Next Investigation
The highest-value next task is:
- inspect official `Detr3DHead` classification-loss normalization and assignment/loss plumbing against our implementation and produce an exact gap list

This is a better next step than making more ad hoc hyperparameter changes.

## Branching Recommendation
Yes, it is a good point to diverge branches now.

Suggested strategy:
- keep current branch as a stable reference branch for the restored encoded-10D baseline
- create separate experiment branches for risky training changes

Suggested branch layout:
1. `baseline/encoded10d-debug`
   - current restored encoded-10D package path
   - should remain readable and minimally changed
2. `exp/loss-normalization-parity`
   - only classification normalization / averaging changes
3. `exp/matcher-parity`
   - only matcher cost implementation changes
4. `exp/head-refine-parity`
   - only head / refinement / decode convention changes
5. `exp/query-count-or-schedule`
   - only query count / training schedule experiments

Reason:
- this project now needs multiple experiments
- it will get hard to reason about results if loss, matcher, head, and schedule changes are mixed together
- branch separation will preserve a clean baseline and make regression tracking much easier

## Branches Created In This Session
All experiment branches were created locally and then repointed to the clean artifact-free commit:
- clean commit: `7c50202` (`Clean tracked artifacts and capture encoded10d debug baseline`)
- `baseline/encoded10d-debug`
- `exp/loss-normalization-parity`
- `exp/matcher-parity`
- `exp/head-refine-parity`
- `exp/query-count-or-schedule`

Intended use:
- `baseline/encoded10d-debug`: stable starting point for the restored encoded-10D path
- `exp/loss-normalization-parity`: compare class-loss normalization against official MMDet / DETR3D behavior
- `exp/matcher-parity`: compare matcher focal-cost / assignment implementation against official behavior
- `exp/head-refine-parity`: compare head channel ordering, decode semantics, and iterative refinement behavior
- `exp/query-count-or-schedule`: query count, schedule, and other training-recipe experiments that should stay isolated from structural parity work

Important note:
- all of those branch refs now point at the same clean local commit `7c50202`
- the repo worktree is clean at the end of this session
- tracked artifacts were deleted and a new `.gitignore` was added for `__pycache__/`, `*.pyc`, and `outputs/`
- remote push could not be verified from this environment because SSH push remained blocked / non-responsive

## Current Local Worktree Status At Session End
Files currently modified in the worktree:
- `SESSION_HANDOFF_2026-03-20.md`
- `detr3d/engine/trainer.py`
- `detr3d/models/heads/detr3d_head.py`
- `detr3d/models/losses/detr3d_loss.py`
- `detr3d/models/losses/loss_utils.py`
- `detr3d/models/losses/matcher.py`
- `notebooks/detr3d_e2e_walkthrough.ipynb`
- plus generated `__pycache__` files

The notebook currently still contains the `num_queries=300` experiment in the main one-sample training cell. That should be reverted first next session before resuming experiments.

## One-Sentence Resume Prompt For Next Codex
Resume from the restored encoded-10D baseline, revert the notebook from `num_queries=300` back to `100`, then compare our classification-loss normalization and matcher implementation line by line against the official DETR3D/MMDet path before making any further training changes.
