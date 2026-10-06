# Submit predictions

The competition requires one combined comma-separated values (CSV) file with
Task 1, Task 2, and Task 4 prediction rows. Task 3 has no separate rows
because it is scored from Task 1.
The [competition and theory guide](COMPETITION_AND_THEORY.md) explains the task
scores and the [trafficbench guide](TRAFFICBENCH.md#build-and-upload-the-competition-file)
shows how to create and validate the combined CSV.

Join the
[Kaggle competition](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview)
to submit. Upload the validated CSV manually on the
[Kaggle submission page](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/submit)
or use the Kaggle command-line interface. Install and authenticate it as
described in [download the data](GET_DATA.md).

~~~sh
uv run kaggle competitions submit \
  -c 2026-ieee-big-data-traffic-flow-bench \
  -f final_submission.csv \
  -m "Baseline submission"
~~~

See the Kaggle CLI's
[competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)
for options such as checking submission status.
