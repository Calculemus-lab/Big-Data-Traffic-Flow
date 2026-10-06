# Traffic Flow Bench team

This repository contains team solutions and a local runner for the 2026 IEEE
Big Data Traffic Flow Bench competition.

## Start here

Follow these guides to understand the tasks, get the data, make predictions,
and submit them.

1. [Competition and traffic theory](docs/COMPETITION_AND_THEORY.md) explains
   what each task predicts, which data solution functions receive, how scores
   work, and the traffic concepts used by the tasks.
2. [Download the data](docs/GET_DATA.md) explains how to join the competition and
   download the release files.
3. [Trafficbench](docs/TRAFFICBENCH.md) explains how to write a solution
   function, run local cases, make final predictions, and assemble the output.
4. [Submit predictions](docs/SUBMIT.md) explains how to upload the assembled
   CSV.

Use the [public release package reference](docs/RELEASE_PACKAGE_REFERENCE.md)
to look up downloaded file paths, table fields, and the fields that identify
each row. You do not need to read it before writing a solution.

## Team workflow

- [Experiment records](docs/EXPERIMENTS.md)
- [Approach notes](docs/APPROACHES.md)
- [Workflow and branch conventions](docs/WORKFLOW.md)
- [Hosted benchmark setup](docs/CI_SETUP.md)
- [Dependencies](docs/DEPENDENCIES.md)
- [Saved results](results/README.md)

The [official competition repository](official_competition_repo/README.md)
contains organizer schemas, scorers, and baselines. It is pinned as a Git
submodule. After cloning this repository, initialize it with:

```sh
git submodule update --init --recursive
```
