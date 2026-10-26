"""Helpers for GitHub Actions workflows."""

import os
from pathlib import Path


def write_environment_variables(values: dict[str, str]) -> None:
    """Append environment variables to the GitHub Actions environment file.

    Later steps in the same job can read the variables.

    Args:
        values: Variable names mapped to their values.

    Raises:
        RuntimeError: If ``GITHUB_ENV`` is not set.
    """
    github_env = os.getenv("GITHUB_ENV")

    if not github_env:
        raise RuntimeError(
            "GITHUB_ENV is not set. "
            "This script must run inside GitHub Actions."
        )

    environment_path = Path(github_env)

    with environment_path.open("a", encoding="utf-8") as environment_file:
        for name, value in values.items():
            environment_file.write(f"{name}={value}\n")
