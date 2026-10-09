# Submit predictions

The competition accepts one combined comma-separated values (CSV) file with
Task 1, Task 2, and Task 4 prediction rows. Task 3 has no separate prediction
rows because the evaluator scores its physical-consistency checks using the
submitted Task 1 values.

First use Trafficbench to create the three task prediction files, combine them
with the competition's submission key and template, and validate the combined
file. The [Trafficbench guide](TRAFFICBENCH.md#build-and-upload-the-competition-file)
shows those commands. Local validation checks the file structure and required
rows; it does not calculate the score against the competition's hidden answers.

Join the
[Kaggle competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
to submit. Upload the validated CSV on the
[competition submission page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/submit)
or use the Kaggle command-line interface. Install and authenticate the CLI as
described in [download the data](GET_DATA.md). From the repository root,
submit the file with:

```sh
uv run --locked kaggle competitions submit \
  2026-ieee-big-data-traffic-flow-bench \
  -f final_submission.csv \
  -m "Baseline submission"
```

See Kaggle's [competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)
for options such as checking submission status. The competition's public
leaderboard uses the validation period. The private period is scored during
the final evaluation and determines the final ranking.
