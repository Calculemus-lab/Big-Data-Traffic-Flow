# Task 4 public validation ablation

Date: 2026-10-16

Each candidate was submitted as a complete 6,980,503-row competition file. Task
1 and Task 2 rows were zero-filled in every candidate, so the public total
isolates Task 4 after the Task 3 zero contribution. The all-zero control had a
public score of 0.00000. Divide the Task 4-only total by its 0.20 competition
weight to estimate the Task 4 score.

| Candidate | Kaggle submission | Public total | Implied Task 4 score |
| --- | ---: | ---: | ---: |
| Original log-space model, 1.5 movement scale | 57147896 | 0.11730 | 0.58650 |
| Regularized least-squares through the baseline runner | 57263945 | 0.07916 | 0.39580 |
| Released weak prior unchanged | 57264038 | 0.10420 | 0.52100 |
| Log-space model, stronger prior regularization | 57264201 | 0.10919 | 0.54595 |
| Original log-space fit, no movement expansion | 57264360 | 0.11748 | 0.58740 |
| Sparse L-BFGS-B quadratic fit, solved to convergence | 57264635 | **0.19874** | **0.99370** |

The top candidate minimizes

```text
0.5 * ||A f - c||² + 0.5 * 0.05 * ||f - b||², subject to f >= 0
```

It starts from the released prior and solves each panel and split independently
with a sparse link-path matrix. The separate project-runner execution produced
70,708 predictions that matched the submitted candidate exactly. Its released
count fit averaged about 0.9999, but that diagnostic alone did not predict the
leaderboard result; the public score came from the hidden path-flow evaluation.

The result establishes a near-perfect Task 4 public score for this validation
month. The private score is not available, and the result does not establish
that other tasks have strong scores.
