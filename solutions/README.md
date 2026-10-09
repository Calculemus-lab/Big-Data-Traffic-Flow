# Solution packages

`baseline` and `example` are the reference implementation and starter example.
The Task 2 experiments live in separate packages so each tested idea can be
run and compared without mixing model changes.

| Package | Role | Current evidence |
| --- | --- | --- |
| [`queue_boosting`](queue_boosting/README.md) | Earlier LightGBM queue model; current official-score reference. | March validation Task 2 IoU: about 0.45047. |
| [`queue_boosting_event_aware`](queue_boosting_event_aware/README.md) | Selector-aware onset classifier and T+30 decoding. | Official validation Task 2 IoU: about 0.71773; current best. |
| [`queue_boosting_event_aware_all_steps`](queue_boosting_event_aware_all_steps/README.md) | Ablation that predicts onset queues at all six horizons. | Official validation Task 2 IoU: about 0.70397. |
The local comparison is summarized in
[`results/queue_boosting/selector_matched_proxy_check.md`](../results/queue_boosting/selector_matched_proxy_check.md).
Local scores use speed-derived proxy labels and do not rank the packages for
the competition. The event-aware package is the current candidate to score on
Kaggle because its onset timing follows the released selector behavior; keep
the previously scored model as the fallback until that comparison is available.
