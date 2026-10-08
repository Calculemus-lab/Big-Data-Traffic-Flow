# GitHub-hosted benchmark experiments

The **Experiment benchmark** workflow is an optional way to run prepared local
train cases on GitHub Actions. It gives the team a shared run environment and
combines results across panels, which is useful when comparing solutions on
different developers' computers. It does not make the competition submission
or replace validation leaderboard results. The current workflow uses standard
CPU runners, so it is intended for CPU experiments rather than GPU training.

Pull request checks are a separate workflow. They run automatically after
changes are pushed to GitHub. Hosted benchmark runs use competition-derived
data and require the one-time private-data setup below.

## Store prepared data privately

The code repository is public. Keep competition data in a separate **private**
GitHub repository as release attachments. Private repositories and releases are
available on GitHub Free. Releases have no total storage/bandwidth limit, and
individual attachments must be under 2 GiB, as described in GitHub's
[release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
The packaging command checks the attachment size.
Do not put benchmark data in Git history, Git Large File Storage (LFS), or
GitHub Actions workflow artifacts.

1. Create a private repository for the team's data, with an initial README commit.
2. Prepare one train interval, then package it locally. The example uses train
   inputs starting November 3, includes historical labels through November 19,
   and generates local targets from November 20 through November 27:

```sh
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-20 \
  --prediction-end-date 2030-11-28 \
  --profile quick --output data/benchmarks/quick
uv run --locked bench package-ci \
  --benchmark data/benchmarks/quick --output data/ci-packages/quick
```

3. Create a release tagged `benchmarks-v1` in the private data repository and
   attach every file produced by the packaging command. You can use GitHub's
   release page or:

```sh
gh release create benchmarks-v1 data/ci-packages/quick/* \
  --repo YOUR_ORG/trafficbench-data --title 'Benchmarks v1' \
  --notes 'Fixed team validation data. Keep this repository private.'
```

4. Add the full profile to the same release when ready:

```sh
uv run --locked bench prepare --data kaggle_public \
  --history-start-date 2030-11-03 \
  --history-end-date 2030-11-20 \
  --prediction-start-date 2030-11-20 \
  --prediction-end-date 2030-11-28 \
  --profile full --output data/benchmarks/full
uv run --locked bench package-ci \
  --benchmark data/benchmarks/full --output data/ci-packages/full
gh release upload benchmarks-v1 data/ci-packages/full/* \
  --repo YOUR_ORG/trafficbench-data
```

The `quick` profile selects one panel and `full` selects every panel. Both
profiles use the single date interval recorded in the prepared benchmark. Do
not replace published assets while experiments are running. Publish a new
release when deliberately changing the prepared interval or case-generation
settings.

## Let the workflow download the data

In the code repository's Actions settings, add:

- Variable `TRAFFICBENCH_DATA_REPO`: the private repository, such as
  `YOUR_ORG/trafficbench-data`.
- Variable `TRAFFICBENCH_DATA_RELEASE`: `benchmarks-v1` (this is also the default).
- Secret `TRAFFICBENCH_DATA_TOKEN`: a fine-grained token with **Contents: read**
  permission on only the private data repository. Metadata read access is implicit.
  Organization owners may need to approve the token. Allow it to last through the
  competition, then revoke it.

The workflow refuses a public data repository. This setup does not create the
private repository, publish data, set secrets or change GitHub billing settings
for you. After setup, teammates do not need credentials on their laptops to
start a benchmark run. Only start runs from trusted team branches because the
workflow can access the download secret.

## What happens in a hosted benchmark run

Open **Actions → Experiment benchmark → Run workflow**. Select the branch,
solution package, task, profile, and parameters. The workflow first downloads
the profile's JSON index, which lists the panel archives. It then starts one
job for each selected panel. Each job downloads that panel's archive, verifies
its SHA-256 checksum, and runs the solution. At most two panel jobs run at a
time. The final job combines the panel results using the same task and panel
weights as local Trafficbench runs. A missing or failed panel result prevents
a combined score.

Each panel job has a 30-minute timeout. The standard private-repository Linux
runner has 8 GB RAM, and the public-repository runner has 16 GB according to the
[GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
Each hosted job scans one panel's prepared tables lazily. A solution reads
only the tables and columns it collects. Solutions that load large histories
into memory may still need to run locally because the hosted job has limited
memory. The workflow's `ubuntu-latest` runner does not provide a GPU. GitHub
offers GPU-enabled larger runners for organizations on eligible plans, but this
repository's workflow is not configured to use one. See GitHub's
[larger-runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners)
if GPU-backed hosted runs become necessary.

The workflow uploads reports, metrics, and a copy of the solution's source
files. It does not upload predictions made from competition data or the
benchmark itself to public GitHub Actions artifacts. Local runs still save
predictions. Keep those files if you need to combine predictions or reproduce
the final submission. Per-panel artifacts expire after one day and combined reports
expire after seven days. Download any results you want to keep longer.

## Free-plan usage

Standard GitHub Actions compute is free and unlimited for this public code repository.
If it becomes private, GitHub Free includes 2,000 minutes per month shared by the
account or organization. Private-repository runs can use that allowance, so favor
the quick profile during development and reserve the full profile for candidate
solutions. Artifact storage on Free is limited to 500 MB. Compact reports and
short retention help stay within that limit.
See GitHub's [Actions billing limits](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
for the current plan allowances.
The workflows do not enable paid runners or change billing budgets.

## Team conventions

Keep each prepared benchmark's date interval and generation settings fixed. To
change them, prepare into a new empty directory and publish a new data release.
Runs record data hashes and the evaluator version. Keep public leaderboard
scores separate from scores on locally generated cases.

Runs save copies of the Python and configuration files and record installed
package versions. For external weights or other assets, record their exact
locations and checksums in your parameters or experiment notes because the run
does not copy them. Before the final submission, preserve the selected model,
predictions, input data, Python version, and package versions. Each experiment
owner generates validation and private predictions.
`bench assemble` and `bench validate` check the resulting submission.

Run project checks locally with `uv run --locked python -m pytest -q`. Branch
protection can make Checks mandatory if your repository settings support it,
but is not required for the workflows themselves.
