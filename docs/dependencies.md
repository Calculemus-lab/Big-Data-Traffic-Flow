# Dependencies

This repository uses [uv](https://docs.astral.sh/uv/) for its shared Python
environment.

```bash
uv sync --locked                   # Set up or update the environment
uv run python path/to/script.py    # Run code in it
uv add <package>                   # Add a dependency
uv remove <package>                # Remove a dependency
```

Always commit `pyproject.toml` and `uv.lock` together. Never edit or hand-merge
`uv.lock`; resolve dependency intent in `pyproject.toml`, then run `uv lock`.
Experiment branches may use different versions, but only shared dependencies
should be merged into `main`.

The `.gitattributes` rule does not block normal `uv.lock` updates. It creates a
conflict only when both branches changed the lockfile; resolve `pyproject.toml`
and regenerate the lockfile with `uv lock`.
