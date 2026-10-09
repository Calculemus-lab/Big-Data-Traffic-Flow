# Validation assessment and template repair

All reported comparisons use the same 80 official March validation windows.
Other task outputs are zero, and the all-zero control scores zero. The Task 2
weight is `0.30`; derived task scores are approximate because Kaggle rounds the
displayed public composite score.

| Method | Training history | Public composite | Task 2 IoU |
| --- | --- | ---: | ---: |
| Persistence | Latest released history reading | 0.09051 | 0.30170 |
| Queue boosting | June 2030 through February 2031 | 0.13514 | 0.45047 |
| Queue boosting | February 2031 only | 0.13032 | 0.43440 |

The nine-month model improves over persistence by approximately 0.14877 queue
IoU, or 49% relative. The 28-day variant is slightly worse. This experiment does
not support training-history length as the main cause of the proxy-to-official
gap. Seeds and sampling differ between the fits, so one comparison does not
fully isolate causality.

Local persistence averaged approximately 0.36 across December and January;
local queue boosting averaged approximately 0.69. The model has a much larger
official-score drop than persistence. Its proxy validation is too optimistic to
use as a direct estimate of competition performance. Recorded-measurement proxy
labels, generated versus official window selection, and the independently
generated March scenario can all affect the comparison. Hidden answers prevent
assigning the discrepancy to one of these factors from aggregate scores alone.

The model may overfit recurring training traffic patterns. No direct use of
prediction-period labels was found: historical cutoffs are tested, and every
official validation history has twelve readings from T-60 through T-5. Original
model and persistence upload values were checked against their prediction files,
with zero mismatches across 174,000 queue rows each.

## Confirmed forecasting weakness

The nine-month model predicts no queue in 14 of the 40 validation onset windows.
The recent-history model does so in 16. Every official window contains a future
queue, so those empty forecasts necessarily score zero. Neither model has empty
ongoing forecasts.

Training includes general sliding histories, including periods with no future
queue. Official forecasts are selected event windows with a guaranteed positive
future. The next experiment should align training and calibration with that
selection, use the provided event information, and decode nonempty queue sets
against window IoU rather than relying only on independent cell thresholds.

Validation should use released train windows where possible, rolling historical
cutoffs, matched deployment settings, and separate onset/ongoing reporting.
Labels remain proxies where official answers are unavailable. More architecture
or parameter searches should follow this evaluation work.

## R20 origin and repair

Fresh Kaggle downloads have correct R1/R2/R3 mask codes. Sixteen local Task 1
templates had R10/R20/R30 values and preserved modification dates of 4 October
2026. The affected files include validation, private, and three train templates.

The official row ends in `R2,,`; the local row ends in `R20,0`, with six fields
under a seven-field header. This is consistent with replacing `,,` with `0,0`
without the separating comma. The exact originating process was not identified.
It is a local rewrite issue; the official downloaded templates are correct.

All sixteen files are restored byte for byte from fresh official downloads.
Malformed originals remain in `temp/queue_input_audit/backups/`. The complete
validation/private state-key set matches the master key with zero missing and
zero extra keys. [The repair record](template_repair.json) includes filenames
and before/after SHA-256 hashes.

This did not invalidate the Task 2 submissions: their keys match the master key,
and other task outputs came directly from it. Queue training restores historical
state values by timestamp, station, and link; the malformed mask code is not used
as a queue feature or label.

## Discarded export

Submission 56918495 came from a one-off builder with validator arguments reversed.
It exported zero target placeholders, and its zero score is excluded. The
corrected file was checked against every intended queue value and submitted as
56918669. The original model and persistence exports used the standard assembler
and have also been checked cell by cell.

[Raw Kaggle records](kaggle_submissions.json),
[comparison results](comparison_results.json),
[original export checks](original_export_verification.json), and
[corrected export checks](recent_export_verification.json) preserve the audit.
Five submissions were made, including the zero control and discarded export.
