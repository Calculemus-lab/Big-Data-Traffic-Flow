# Traffic Flow Bench team experiments

Team experiment code for the 2026 IEEE Big Data Traffic Flow Bench.

- [Getting started and using `bench`](docs/GETTING_STARTED.md)
- [CI and private data setup](docs/CI_SETUP.md)
- [Competition reference](docs/COMPETITION.md)
- [Experiment records](docs/EXPERIMENTS.md)

The checked-in [public benchmark reference](trafficflowbench-public/README.md)
contains the organizer's schemas, scorers, and baseline documentation.

The goal is to maximize local cross-validation score first, then public
leaderboard score.

Additional project guides cover:

- [Competition and traffic theory](docs/competition_and_theory.md)
- [Data download](docs/data.md)
- [Submissions](docs/submit.md)
- [Workflow and branch conventions](docs/workflow.md)
- [Python dependencies](docs/dependencies.md)
- [Result artifacts](results/README.md)

The official competition repository is also pinned as a submodule in
[`official_competition_repo/`](official_competition_repo/). Initialize it after
cloning with `git submodule update --init --recursive`, or clone with
`--recurse-submodules`.
