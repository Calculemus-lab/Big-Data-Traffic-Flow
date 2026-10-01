# Big-Data-Traffic-Flow

Shared repository for experiments in the Kaggle **2026 IEEE Big Data Traffic Flow Bench** competition.

Competition: https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/overview

The goal is to maximize local cross-validation score first, then public
leaderboard score.

### Project guides:    
- [competition and theory](docs/competition_and_theory.md)
- [data](docs/data.md)  
- [submission](docs/submit.md)
- [workflow and branch conventions](docs/workflow.md)    
- [Python dependencies](docs/dependencies.md)   
- [experiment logging](experiments/README.md)     
- [result artifacts](results/README.md)     


The official competition code, baselines, evaluators, and reference docs are
pinned in [`official_competition_repo/`](official_competition_repo/). Initialize
the submodule after a regular clone with `git submodule update --init --recursive`,
or clone this repository with `--recurse-submodules`.
