# Focused Task 2 proxy comparison

Four current solution variants were run on the same prepared December case for
`D12_I5_N`: five locally generated onset windows and five ongoing windows. The
prepared benchmark ID is
`5c2b706b633167bb26eb9d415ae1fe248c4b98c97c0471794fc3d71a7f4b5c37`. Metrics
are mean per-window IoU against local speed-threshold proxy labels.

| Variant | All windows | Onset | Ongoing |
| --- | ---: | ---: | ---: |
| Original horizon model | 0.671153 | 0.499057 | 0.843249 |
| Event-aware, T+30 only | 0.589187 | 0.335471 | 0.842903 |
| Event-aware, all six onset steps | **0.696647** | **0.550392** | 0.842903 |
| Event-aware with three-hour context | 0.576542 | 0.329019 | 0.824065 |

The all-step ablation wins this local case because it can match proxy queues
that start before T+30. Three-hour context does not help here. This does not
justify removing the T+30 rule from the competition candidate: the organizer's
selector chooses official onset windows differently from this proxy sampler.
The rule is supported by the released official train-window timing and should
be judged on an official score.

I also checked whether this date range contained local onset windows with no
proxy queue through T+25 and a queue at T+30. It had none for `D12_I5_N` from
December 2030 through February 2031, so the local proxy data cannot support a
selector-matched comparison for that interval.

The metric table is kept in
[`selector_matched_proxy_check.csv`](selector_matched_proxy_check.csv). The
run snapshots for the three retained model variants remain under the ignored
`runs/` directory for inspection.
