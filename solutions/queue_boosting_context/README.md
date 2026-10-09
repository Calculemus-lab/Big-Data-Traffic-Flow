# Queue forecasting with causal three-hour context

This experimental variant keeps the event-aware LightGBM design and adds
summaries of the three hours immediately before each standard 60-minute history.
At prediction time it reads the selected split's released mainline and ramp
records only in `[T-4 hours, T)`. Missing or masked observations stay missing.

The summaries cover speed relative to free-flow speed, flow relative to
capacity, occupancy, attached ramp flows, observation coverage, and queue share.
The existing model continues to use the detailed readings from `[T-60 minutes,
T)`. Onset forecasts remain restricted to T+30.

Run the small prepared proxy fold used to compare this variant with the
event-aware model:

```sh
uv run --locked bench experiment queue_boosting_context --task queue \
  --benchmark data/benchmarks/queue-context-december-quick
```

Local queue labels are speed-derived proxies. Compare this variant with
`queue_boosting_event_aware` on identical prepared windows, especially for
ongoing queues. These results do not replace an official validation score.
