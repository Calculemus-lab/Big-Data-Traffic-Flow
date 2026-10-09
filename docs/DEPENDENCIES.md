# Dependencies

The repository uses [uv](https://docs.astral.sh/uv/) to create its Python
environment and keep installed packages consistent with the dependency files.
`pyproject.toml` declares the project and development dependencies;
`uv.lock` records the resolved package versions. The project requires Python
3.12 and pins the uv version in `pyproject.toml`.

Run commands from the repository root:

```sh
uv sync --locked                   # Install the versions recorded in uv.lock
uv run --locked python path/to/script.py
uv add <package>                   # Add a project dependency
uv remove <package>                # Remove a project dependency
```

`uv add` and `uv remove` update the dependency declaration and lockfile. Commit
`pyproject.toml` and `uv.lock` together. Do not edit or hand-merge `uv.lock`;
when dependency declarations need to be resolved again, update
`pyproject.toml` and run `uv lock`.

Experiment branches may use different dependency versions. Keep a dependency
change on the branch that needs it, and resolve the lockfile from that
branch's `pyproject.toml` if branches with different dependencies are
combined. The repository's `.gitattributes` settings allow ordinary lockfile
updates; a conflict can still occur when both branches change `uv.lock`.
