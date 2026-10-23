#!/usr/bin/env bash
# Run the CAL-31 study and later-month assessment on the university CPU server.
set -euo pipefail

release_directory="${1:?Provide the extracted release directory}"
work_directory="${2:?Provide the experiment artifact directory}"
project_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$work_directory/tmp" "$work_directory/cache"
cd "$project_directory"

# Keep every generated artifact and package cache in the requested data storage.
export TMPDIR="$work_directory/tmp"
export XDG_CACHE_HOME="$work_directory/cache"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=8
export POLARS_MAX_THREADS=4
export NUMEXPR_NUM_THREADS=1
export CAL31_RELEASE_DIRECTORY="$release_directory"
export CAL31_WORK_DIRECTORY="$work_directory"

# One controller owns the study; children inherit a 128 GiB address-space ceiling.
exec 9>"$work_directory/pipeline.lock"
flock --nonblock 9
ulimit -v 134217728

# Prepare separate tuning and later-month cases. Existing complete cases are verified.
.venv/bin/python -u - <<'PY'
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from trafficbench.contracts import ALL_PANELS
from trafficbench.prepare import load_config, prepare
from trafficbench.runner import verify_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
release = Path(os.environ["CAL31_RELEASE_DIRECTORY"])
work = Path(os.environ["CAL31_WORK_DIRECTORY"])
scheme = Path("config/queue_selector_matched.yaml")
config = load_config(scheme)
panels = [panel for panel in ALL_PANELS if panel not in config.queue.excluded_panels]
futures = []
with ThreadPoolExecutor(max_workers=2) as executor:
    for name, cutoff, end in [
        ("december", date(2030, 12, 1), date(2031, 1, 1)),
        ("february", date(2031, 2, 1), date(2031, 3, 1)),
    ]:
        destination = work / "benchmarks" / name
        if (destination / "manifest.json").exists():
            verify_benchmark(destination)
            continue
        futures.append(executor.submit(
            prepare, release, destination, date(2030, 6, 1), cutoff, end,
            profile="full", benchmark_scheme_file=scheme, panels=panels,
        ))
    for future in futures:
        print("Prepared:", future.result(), flush=True)
PY

# Use at most 16 concurrent trials with eight model threads per trial.
.venv/bin/python -u -m scripts.optimize_queue \
  --benchmark "$work_directory/benchmarks/december" \
  --output "$work_directory/optuna" \
  --trials 32 --jobs 16 --threads 8

# Assess the frozen selection on February without changing the selected trial.
.venv/bin/python -u - <<'PY'
import json
import logging
import os
from pathlib import Path

from scripts.tune_queue import evaluate
from solutions.queue_boosting_event_aware.model import QueueParameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
work = Path(os.environ["CAL31_WORK_DIRECTORY"])
parameters = QueueParameters.model_validate_json(
    (work / "optuna" / "selected_params.json").read_text()
)
benchmark = work / "benchmarks" / "february"
baseline = evaluate(QueueParameters(threads=8), benchmark, work / "assessment")
candidate = evaluate(parameters, benchmark, work / "assessment")
result = {
    "reference": baseline.model_dump(mode="json"),
    "candidate": candidate.model_dump(mode="json"),
    "gain": candidate.score - baseline.score,
}
(work / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
PY
