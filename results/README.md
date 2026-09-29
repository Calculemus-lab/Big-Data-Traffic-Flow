# Results

Keep persistent experiment outputs in `results/` on that experiment's branch.
This includes graphs, CSVs, out-of-fold and test predictions, and model
weights. Record prediction paths and run details in
`experiments/experiment_log.csv`.

Save out-of-fold and test predictions for each experiment, along with the
configuration or notes needed to reproduce them. Use the `oof_path` and
`test_path` fields in the experiment log for their locations.

Use `temp/` directories for disposable intermediate files instead.
