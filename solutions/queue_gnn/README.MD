# queue_gnn: graph network for queue forecasting (Task 2)

CAL-16, branch `cycle-1/CAL-16/queue-gnn`. A first learned model for Task 2, kept small so
it trains in about a minute on a laptop CPU.

## How it works

- Each link gets 45 inputs: the last 60 minutes (12 steps) of speed relative to free speed,
  occupancy and a "was observed" flag, the time of day and a weekend flag, and the link's
  usual queue rate for the next six 5-minute slots, taken from the training data.
- A small network encodes each link, then passes information between neighbouring links
  along the corridor, with separate weights for downstream and upstream neighbours.
- The output is a queue probability for each link at T+5 to T+30. A link is predicted
  queued when the probability is at least 0.4.
- Training uses every 5-minute origin in the training period, 4 epochs, with queued cells
  weighted 3x. Cells without a reliable observation are left out of the loss.

```sh
bench run queue_gnn --task queue --params '{"layers": 6}'
```

## Quick results (D12_I5_N, first week of December, benchmark 5c9a87ded0a3)

| Run | Ongoing | Onset | Proxy IoU |
| --- | ---: | ---: | ---: |
| baseline (persistence) | 0.549 | 0.000 | 0.2743 |
| queue_gnn, 0 layers (no neighbours) | 0.577 | 0.181 | 0.3791 |
| queue_gnn, 2 layers (default) | 0.566 | 0.195 | 0.3805 |
| **queue_gnn, 6 layers** | 0.633 | 0.161 | **0.3968** |


## What I learned

- Almost all of the gain over the baseline comes from onset windows. Persistence scores 0
  there by construction; the model reaches about 0.16 to 0.20.
- With 0 or 2 layers the score is nearly the same, so at that depth the gain comes from the
  per-link features, not from the graph.
- Six layers help ongoing queues (0.566 to 0.633) but not onsets (0.195 to 0.161).

## Caveats and next steps

- The score is a proxy: labels come from released speeds, not the hidden traffic state.
- One corridor, one week, 10 windows and one seed per setting, so small differences may be
  noise.
- Next: run on all corridors, submit to Kaggle for a real
  score, then tune (threshold, class weight, depth, epochs) with several seeds.