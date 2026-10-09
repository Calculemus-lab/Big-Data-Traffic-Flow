# Event-aware model proxy checks

These December and January runs compare the selector-aware onset head against
the earlier horizon model on the same 160 local proxy windows. The local
benchmark samples onset windows at random from any forecast with a future
queue. The competition selector instead chooses the earliest origin whose
horizon contains the queue. These scores therefore test generalization to the
local proxy cases, not the selector-matched official onset distribution.

| Condition | Earlier model | Event-aware model | Change |
| --- | ---: | ---: | ---: |
| Onset | 0.516876 | 0.282878 | -0.233998 |
| Ongoing | 0.868108 | 0.872589 | +0.004481 |
| All windows | 0.692492 | 0.577733 | -0.114759 |

The drop is concentrated in onset windows; ongoing performance is nearly
unchanged. This is a negative result for the random-proxy distribution, but it
does not settle performance on official selector-matched onset windows. The
official candidate predictions and submission-limit outcome are documented in
[`../validation/event_aware/README.md`](../validation/event_aware/README.md).

Each month directory keeps per-window metrics, the run record, and the exact
benchmark manifest. Transient prediction copies and duplicate persistence
outputs were removed.
