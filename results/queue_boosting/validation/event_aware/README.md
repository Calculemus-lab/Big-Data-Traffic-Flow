# Selector-aware Task 2 validation candidate

The event-aware forecaster used history from 2030-06-01 through 2031-03-01
(exclusive) and predicted all official validation and private Task 2 rows. The
saved file contains 174,000 rows for 160 windows. Its run record is in
[`run.json`](run.json).

TrafficFlowBench assembled and validated a complete upload with all
6,980,503 official submission rows. State and ODME rows were zero; only queue
predictions differed. Kaggle accepted the Task 2 candidate as submission
`57007109`, which scored `0.21532` on the public leaderboard. This corresponds
to approximately `0.71773` Task 2 IoU because all other task rows were zero.

The prediction file SHA256 is
`5745e3b79a269ed2baf0bafb3daac5830eddedd7fdcd184e2b9d26cee0f3bd5a`.
The all-horizons ablation scored `0.21119` composite (`0.70397` approximate Task
2 IoU) on the same validation split. The T+30 decoder is the stronger of these
two official candidates. Their difference is modest; private scores remain
hidden.
