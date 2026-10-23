# CAL-31 university-server follow-up

Research date: 23 October 2026. This follow-up used Drogium and the release
prepared for the CAL-31 Task 2 experiments.

## Server and data

The repository clone is at `/local/s4501020/Big-Data-Traffic-Flow`. The Kaggle
release was downloaded directly from Kaggle's storage to
`/data/s4501020/trafficflow/release.zip`, extracted to
`/data/s4501020/trafficflow/kaggle_public`, and opened successfully by
TrafficBench. The archive is about 8.34 GB and the extracted release is about
12.39 GB. TrafficBench accepts this data path directly; the repository does
not need a data symlink. Run artifacts, benchmark preparations, package caches,
and temporary files were kept under `/data/s4501020/trafficflow`.

## Optuna search

The server run used Optuna 5.0 with a seeded TPE sampler, SQLite storage, four
queued reference and ablation trials, and panel-level pruning. It varied onset
tree size, minimum leaf size, class weight, iteration count, missing-history
dropout, CatBoost blend weight, and ongoing-event training scope. It scored the
TrafficBench balanced window IoU on eight panels in the December benchmark.

The intended 32-trial batch was stopped early after 17 trials completed and 15
failed or were interrupted. The 128 GiB virtual-address-space limit caused
allocation failures even though Drogium had ample physical memory. The remaining
large-panel fits were too slow for the requested turnaround, so they were not
restarted. The study database and per-panel run artifacts remain on the server.

| Candidate | December balanced IoU | February check |
| --- | ---: | ---: |
| Reference | 0.888597 | 0.812053 across two panels |
| Best completed Optuna trial 5 | 0.910235 | 0.820514 on D12_I5_N |
| Dropout ablation, trial 1 | 0.904691 | 0.812053 across two panels |

Trial 5 used 63 onset leaves, 30 minimum rows per leaf, and restricted horizon
training to ongoing-event windows. It scored 0.015488 below the reference on
the first February panel, so the December gain did not transfer there. The
dropout ablation tied the reference on both February panels.

The earlier six-trial LightGBM search had selected 63 leaves, 30 minimum rows,
positive weight 4, and feature fraction 0.75. It scored 0.870681 on the January
holdout versus 0.861770 for the reference. On the first February panel it scored
0.779945 versus 0.836002 for the reference. This later result also indicates
that the January gain was not reliable across months.

These February checks cover only D12_I5_N and D7_I10_E, not the full eight-panel
benchmark. They are sufficient to reject the tested candidates for immediate
submission, but they do not establish which model is best across all panels or
the competition's hidden labels. At that stage, no new Kaggle submission had
been made because the tested candidates failed to improve the later-month
reference check.

## Official Kaggle probe

The full-panel December comparison later selected a shared-horizon model with
63 leaves, 30 minimum rows per leaf, positive weight 4, and feature fraction
0.75. It scored 0.901516 on December, compared with 0.888597 for the reference,
but scored 0.843952 on February, compared with 0.854354 for the reference. This
mixed local evidence motivated one official public-validation probe; it was not
treated as proof of improvement.

Kaggle submission 57561677 was generated from the full released training period
and covers every validation and private Task 2 window. Task 1 and Task 4 values
are zero, matching the earlier Task 2-only uploads. The assembled file passed
TrafficBench validation with 6,980,503 rows. Its SHA-256 is
`91c8a991f64a97815bf3afd5b4942926a21db831fb2d4885988c906cbed2efe1`; the file
and prediction run remain under
`/data/s4501020/trafficflow/experiments/cal31/kaggle-probe/` on Drogium.

The public composite score is 0.21169, below the previous best Task 2-only
submission 57007109 at 0.21532, and slightly above the all-horizons ablation
57007717 at 0.21119. Because the other tasks are zero, these correspond to
approximate Task 2 scores of 0.70563, 0.71773, and 0.70397. Kaggle reports no
private score before final evaluation. The probe therefore confirms that this
candidate is not the best public Task 2 submission, while its private ranking
remains unknown.

## Reproduction artifacts

The complete Optuna database, selections, trial table, model runs, and partial
February checks are under
`/data/s4501020/trafficflow/experiments/cal31/` on Drogium. The server runner
was stopped after the partial study; no training or prediction process remains
active.
