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
the competition's hidden labels. No new Kaggle submission was made because the
tested candidates failed to improve the later-month reference check. The
previous Task 2-only Kaggle record remains the reference.

## Reproduction artifacts

The complete Optuna database, selections, trial table, model runs, and partial
February checks are under
`/data/s4501020/trafficflow/experiments/cal31/` on Drogium. The server runner
was stopped after the partial study; no training or prediction process remains
active.
