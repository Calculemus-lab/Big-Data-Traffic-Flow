# Task 2 Queue Forecasting: Progress Report

## Training data and features

One model per panel (8 panels with Task 2; D12\_I405\_N/S have none), trained on every possible window rather than only the official ones.

- **Source:** masked mainline states of the train split, averaged over stations to one value per link and 5-minute step.
- **Label:** a link is queued when speed ≤ 0.60 × free-flow speed. It is labelled only where all stations on the link have speed and ≥ 75% coverage.
- **Examples:** every origin with 12 history and 6 forecast steps and at least one queued cell in the forecast. Non-ongoing (onset-type) windows get loss weight 3.
- **Features (45 per link):** 12 history steps × (speed ratio, occupancy, known flag) = 36; clock sin, cos and weekend = 3; usual queue rate for the 6 forecast time slots = 6.
- **Known mismatch:** training uses masked states (\~61% known cells), while validation and private histories look less masked (\~76%).

## Model and hyperparameters

QueueSTGNN, a spatio-temporal graph network. It treats the 12 history steps as a sequence and the road as a graph, in each block:

- **Time:** a causal 1D convolution (kernel 3) over each link's history, so a step only sees earlier steps.
- **Space:** each link is mixed with the links ahead and behind it (row-normalised adjacency from the LWR mainline topology), followed by a residual connection and layer norm.
- **Output:** the last step's state, the time-mean of the states and an encoding of the static features (clock, usual queue rate) are joined and mapped to 6 queue probabilities.

| Setting | Value |
| --- | --- |
| Step embedding | Linear 3→48 per history step (speed ratio, occupancy, known flag) |
| Blocks | 2 spatio-temporal blocks, 48 channels |
| Static branch | Linear 9→48, ReLU (clock 3 + usual queue rate 6) |
| Head | Linear 144→48, ReLU, Linear 48→6, sigmoid |
| Loss | Weighted BCE, positive weight 3, onset weight 3 |
| Optimiser | Adam, lr 2e-3, batch 64, 6 epochs, seed 0 |
| Threshold | 0.4 |
| Onset rule | Last run: onset\_zero\_steps = 5 (onset windows predict no queue at steps 1–5) |

**Previous architecture (QueueGCN):** the same graph mixing, but the 12 history steps were flattened into the 45 input features of a per-link MLP (45→64→64, then 2 graph layers, then 64→6). It had no temporal modelling; time entered only as flat features. It gave the Kaggle IoU of about 0.37.

## Validation set

The held-out official training windows: 10 per panel (5 onset, 5 ongoing), 80 in total, between 1 and 20 June 2030 (most before 8 June; D7\_I10\_W runs to 20 June). Each panel's model is trained only on data after its last official window, so these windows are unseen.

- **Metric:** IoU per condition (onset, ongoing), averaged window → panel → freeway family → overall. A window with nothing queued in both truth and prediction scores 1.
- **Scripts:** `holdout.py` saves the probabilities per panel; `threshold_ablation.py` scores threshold variants without retraining.
- **Limit:** about 40 windows per condition, so differences under \~2 standard errors are noise.

## Threshold ablation

Predicting onset only at step 6 (nothing at steps 1–5, where no queue exists) lifts the holdout score from 0.50 to 0.70. After that, no threshold variant clearly beats fixed 0.4. Thresholds are fitted on all days but one and scored on the held-out day (out of fold); ± is the standard error of the gain over fixed 0.4.

| Onset rule | Variant | Params | Onset IoU | Ongoing IoU | Out-of-fold score | vs fixed 0.4 (± SE) |
| --- | --- | --- | --- | --- | --- | --- |
| Model at all 6 steps | Fixed 0.4 | 0 | 0.181 | 0.823 | 0.502 | +0.000 |
| Model at all 6 steps | Global | 1 | 0.230 | 0.823 | 0.527 | +0.025 ± 0.023 |
| Model at all 6 steps | Per condition | 2 | 0.230 | 0.827 | 0.529 | +0.027 ± 0.022 |
| Model at all 6 steps | Per condition × step | 12 | 0.605 | 0.807 | 0.706 | +0.204 ± 0.036 |
| Model at all 6 steps | Per panel × condition | 16 | 0.311 | 0.814 | 0.562 | +0.061 ± 0.031 |
| Onset at step 6 only | Fixed 0.4 | 0 | 0.583 | 0.823 | 0.703 | +0.000 |
| Onset at step 6 only | Global | 1 | 0.602 | 0.816 | 0.709 | +0.006 ± 0.028 |
| Onset at step 6 only | Per condition | 2 | 0.621 | 0.827 | 0.724 | +0.021 ± 0.025 |
| Onset at step 6 only | Per condition × step | 12 | 0.621 | 0.807 | 0.714 | +0.011 ± 0.025 |
| Onset at step 6 only | Per panel × condition | 16 | 0.616 | 0.814 | 0.715 | +0.012 ± 0.033 |

With the onset rule on, the single-threshold sweep peaks at 0.30 (onset 0.621, ongoing 0.819), but 0.325 drops to onset 0.557, so that peak is a spike. Thresholds 0.45–0.55 form a plateau (average about 0.71), and 0.4 scores 0.703. Per-step fitting selects "never predict" for onset steps 1–5 (0.05 is equivalent once the rule is on) and 0.30 for step 6. Chosen: one global threshold of 0.4. The persistence baseline scores onset 0.025, ongoing 0.590, 0.307 overall.

## Final results

| Model | Threshold | Onset rule | Holdout onset | Holdout ongoing | Holdout score | Kaggle IoU                                 |
| --- | --- | --- | --- | --- | --- |--------------------------------------------|
| Persistence | n/a | n/a | 0.025 | 0.590 | 0.307 | 0.302                                      |
| GCN&#32; | 0.4 | model all 6 steps |  |  |  | 0.410                                      |
| GCN&#32; | 0.4 | onset at step 6 only |  |  |  | 0.551                                      |
| Spatio-temporal GNN | 0.4 | model at all 6 steps | 0.181 | 0.823 | 0.502 | -                                          |
| Spatio-temporal GNN | 0.4 | onset at step 6 only | 0.583 | 0.823 | 0.703 | 0.56 (0.42 for onset and 0.69 for ongoing) |
| Spatio-temporal GNN, global threshold fitted out of fold | fitted | onset at step 6 only | 0.602 | 0.816 | 0.709 | -                                          |

## Future steps
