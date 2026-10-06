"""Check repository conventions that Git alone cannot enforce."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

BRANCH_PATTERN = re.compile(r"cycle-[0-9]+/CAL-[0-9]+/[a-z0-9]+(?:-[a-z0-9]+)*\Z")
# Existing setup branches predate the experiment convention.
LEGACY_BRANCHES = {"CAL-7/setup-repo", "chore/setup-CV"}
SUBMODULE_ROOTS = {"official_competition_repo"}
REFERENCE_ROOTS = {"trafficflowbench-public"}
ARTIFACT_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".html",
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".safetensors",
}
ARTIFACT_EXCEPTIONS = {"experiments/experiment_log.csv"}
DOCUMENTATION_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}
IMAGE_ARTIFACT_EXEMPT_DIRECTORIES = {"docs"}


def main() -> int:
    """Check branch names and tracked artifact locations; return a shell status."""
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    errors: list[str] = []
    if (
        branch
        and branch != "main"
        and branch not in LEGACY_BRANCHES
        and not BRANCH_PATTERN.fullmatch(branch)
    ):
        errors.append(f"branch {branch!r} must match cycle-#/CAL-#/experiment-name")

    tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    for name in filter(None, tracked):
        path = PurePosixPath(name)
        if path.parts[0] in SUBMODULE_ROOTS:
            continue
        if "temp" in path.parts[:-1]:
            errors.append(f"tracked temporary file: {name}")
        if path.suffix.lower() == ".ipynb":
            errors.append(f"tracked notebook: {name}")
        if (
            path.suffix.lower() in ARTIFACT_SUFFIXES
            and name not in ARTIFACT_EXCEPTIONS
            and path.parts[0] != "results"
            and path.parts[0] not in REFERENCE_ROOTS
            and not (
                path.parts[0] in IMAGE_ARTIFACT_EXEMPT_DIRECTORIES
                and path.suffix.lower() in DOCUMENTATION_IMAGE_SUFFIXES
            )
        ):
            errors.append(f"experiment artifact must be under results/: {name}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Repository conventions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
