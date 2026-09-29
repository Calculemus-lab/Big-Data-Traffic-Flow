## Summary

<!-- What changed, and why? -->

## Experiment checklist

- [ ] Branch follows `cycle-#/CAL-#/experiment-name` and contains one experiment.
- [ ] Every run, including failures, is recorded in `experiments/experiment_log.csv`.
- [ ] Persistent outputs are in `results/`; temporary files are in `temp/`.
- [ ] Code is in Python files and uses the shared CV scheme.

## Dependency checklist

- [ ] I ran `uv lock --check` and `uv sync --locked`.
- [ ] If dependencies changed, I committed `pyproject.toml` and `uv.lock` together.
- [ ] I regenerated `uv.lock` with uv instead of editing or hand-merging it.
