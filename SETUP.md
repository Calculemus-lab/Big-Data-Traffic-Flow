# Owner setup: GitHub Free

Both workflows use standard GitHub-hosted Ubuntu machines. Nobody needs to
register a runner, rent a server, or leave a computer switched on.

PR checks work as soon as the workflow files are pushed to GitHub. Real-data
benchmarks need one-time data provisioning, described below.

## Store prepared data privately

The code repository is public. Keep competition data in a separate **private**
GitHub repository as release attachments. Private repositories and releases are
available on GitHub Free. Releases have no total storage/bandwidth limit, and
individual attachments must be under 2 GiB. The packaging command checks this.
Do not put benchmark data in Git history, Git LFS, or Actions artifacts.

1. Create a private repository for the team's data, with an initial README commit.
2. Package a prepared benchmark on your laptop:

```sh
bench package-ci --bundle data/benchmarks/quick --output data/ci-packages/quick
```

3. Create a release tagged `benchmarks-v1` in the private data repository and attach
   every file produced by that command. You can use GitHub's release page or:

```sh
gh release create benchmarks-v1 data/ci-packages/quick/* \
  --repo YOUR_ORG/trafficbench-data --title 'Benchmarks v1' \
  --notes 'Fixed team validation data; keep this repository private.'
```

4. Add the full profile to the same release when ready:

```sh
bench package-ci --bundle data/benchmarks/full --output data/ci-packages/full
gh release upload benchmarks-v1 data/ci-packages/full/* --repo YOUR_ORG/trafficbench-data
```

The holdout profile uses the same process after preparing it locally. Do not
replace published assets while experiments are running. Publish a new release
when deliberately changing the validation scheme.

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
for you. No credentials are needed on teammates' laptops to trigger CI runs once
this is configured. Trigger only trusted team branches because manual workflows
can access the download secret.

## What happens in CI

Open **Actions → Experiment benchmark → Run workflow**. Pick the branch, solution,
task, profile and parameters. GitHub downloads the small index, then starts a job
for each applicable road panel. Each job downloads only that panel's data, checks
its checksum and runs the experiment and baseline. Two panel jobs run at a time.
The final job combines all panel results using the same weighting as local runs.
A missing or failed panel prevents a combined score.

Each panel job has a 30-minute timeout. The standard private-repository Linux
runner has 8 GB RAM; public-repository runners have 16 GB. The training loader uses
compact Arrow storage to reduce memory usage. Arbitrary large models may still
need to be run locally; model resource needs are not guaranteed by the harness.

Reports, metrics and source snapshots are uploaded. Real-data predictions and the
benchmark itself are not uploaded to public Actions artifacts. Predictions are
still saved on local runs. Keep local runs for blending and final reproducibility.
Intermediate panel artifacts expire after one day and combined reports after seven
days. Download results you want to keep longer.

## Free-plan usage

Standard Actions compute is free and unlimited for this public code repository.
If it becomes private, GitHub Free includes 2,000 minutes per month shared by the
account/organization. Private-repository runs can exhaust that allowance, so favor
quick runs during development and full runs for candidates. Artifact storage on
Free is limited to 500 MB; compact reports and short retention help stay within it.
The workflows do not enable paid runners or change billing budgets.

GitHub references:
- https://docs.github.com/en/actions/reference/limits
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases

## Team conventions

Keep the fold configuration fixed. To change it, use a new scheme name and prepare
into a new empty directory. Runs record data hashes and the evaluator version.
Use February sparingly, and keep leaderboard results separate from local proxies.

Runs snapshot Python/configuration sources and record installed package versions.
For external weights or assets, save their immutable locations/checksums in your
parameters or experiment notes; they are not automatically copied. Before the
final submission, preserve the selected model, predictions, data and environment.
Generating validation/private predictions remains the experiment author's job;
`bench assemble` and `bench validate` check the resulting submission.

Run project checks locally with `python -m pytest -q`. Branch protection can make
Checks mandatory if your repository settings support it, but is not required for
the workflows themselves.
