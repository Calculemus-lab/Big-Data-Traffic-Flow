# Selector-aware Task 2 validation candidate

The event-aware forecaster used history from 2030-06-01 through 2031-03-01
(exclusive) and generated all official validation and private Task 2 rows.
The saved prediction file contains 174,000 rows for 160 windows. Its run
record is in [`run.json`](run.json).
That run record preserves the original CLI solution name; the same candidate
code is now isolated under `solutions/queue_boosting_event_aware`.

TrafficFlowBench assembled and validated a complete upload with all
6,980,503 official submission rows. State and ODME rows were zero; only queue
predictions differed. Kaggle rejected the submission request with HTTP 400.
The competition's submission-limit endpoint confirmed five submissions that
day and zero remaining. The candidate was not accepted, so there is no public
score for it. The assembled 199 MB upload and generated zero files were
removed; they can be recreated from the saved Task 2 predictions and official
template.

The prediction file SHA256 is
`5745e3b79a269ed2baf0bafb3daac5830eddedd7fdcd184e2b9d26cee0f3bd5a`.
The rejected assembled upload SHA256 was
`1b951e6063d3241418b9c2ae3c0bfaf41e50e7115025d25dfc4868f6f6332498`.

The two-month proxy comparison in
[`../../event_aware_proxy/README.md`](../../event_aware_proxy/README.md)
shows weaker onset results on randomly selected local windows and nearly
unchanged ongoing results. Because those proxy windows do not match the
competition's earliest-event selector, the candidate needs an official score
before it should replace the prior submission-tested model.
