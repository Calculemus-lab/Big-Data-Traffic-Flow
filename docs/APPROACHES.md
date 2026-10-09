# Candidate approaches

Start with the published baseline for each task. Compare candidate methods on
the same prepared train cases so that a score change reflects the method,
rather than a different date range or set of panels. Local Task 2 proxy scores
and generated Task 4 scores are useful for comparing candidates, but they do
not guarantee the same ranking against the competition's hidden answers. See
the [task definitions and scoring guide](COMPETITION_AND_THEORY.md) and the
[local scoring guide](TRAFFICBENCH.md#prepare-and-score-local-train-cases).

The ideas below are starting points for experiments, not claims that a method
will perform well on this release.

## Methods that use the traffic network

- Model traffic as a continuous flow, or represent individual vehicle
  movements. These views can help express flow conservation and congestion
  propagation.
- Use graph neural networks to share information between connected road links.
  Examples include graph convolutional networks, graph attention networks,
  and GraphSAGE.

## Task 1: traffic-state reconstruction

- Estimate missing speed and flow from nearby times, nearby detectors, and
  repeating weekday or time-of-day patterns. Statistical tools include
  Bayesian estimation and Kalman filtering.
- Apply traffic-flow models such as Lighthill-Whitham-Richards (LWR) or
  Aw-Rascle-Zhang (ARZ) to enforce plausible propagation and conservation.
- Compare recurrent neural networks, including long short-term memory (LSTM)
  and gated recurrent units (GRU), with attention-based sequence models and
  tree-based methods such as random forests or gradient boosting.

## Task 2: short-term queue forecasting

- Use a persistence baseline that repeats the latest observed queue state, or
  a moving average of recent traffic measurements.
- Compare statistical time-series models, recurrent neural networks, and
  attention-based forecasting models.
- Use the queue-onset and ongoing-queue conditions separately when diagnosing
  local scores; the official score gives them equal weight.

## Task 3: physical consistency

- Adjust Task 1 predictions to fit the per-lane fundamental diagram and the
  link-level conservation equation. Task 3 scores these physical properties
  from the submitted Task 1 values; it does not have a separate prediction
  function.

## Task 4: origin-destination path-flow estimation

- Fit nonnegative path flows to observed link counts while regularizing the
  estimates toward the released Weak Prior.
- Compare linear programming, convex optimization, and other constrained
  estimation methods. Evaluate both link-count fit and, on generated local
  cases, accuracy against the known generated path flows.

## External examples

These public Kaggle notebooks describe possible traffic-flow approaches:

- [Physics-pipeline submission](https://www.kaggle.com/code/soukeaizenz/0-809-lb-top-50-open-sourcing-physics-pipeline)
- [Traffic-flow pipeline](https://www.kaggle.com/code/lamhuy8904/traffic-flow-bench-pipeline)
- [Hydrodynamic reconstruction, queue, and ODME](https://www.kaggle.com/code/avikdas567/hydrodynamic-traffic-reconstruction-queue-odme)

Use them as reading material. Their reported leaderboard results do not by
themselves show how a method will behave on the data and scoring rules in this
repository.
