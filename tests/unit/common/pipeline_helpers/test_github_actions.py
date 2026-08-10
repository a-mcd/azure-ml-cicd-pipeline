"""Unit tests for GitHub Actions helpers."""

import pytest

from src.common.pipeline_helpers.github_actions import write_environment_variables


def test_write_environment_variables_writes_all_values(
    monkeypatch,
    tmp_path,
):
    # Purpose: Check that each supplied value is written using the GitHub
    # Actions environment-file format.
    github_env = tmp_path / "github_env"

    monkeypatch.setenv("GITHUB_ENV", str(github_env))

    write_environment_variables({
        "MODEL_VERSION": "3",
        "STORAGE_TYPE": "datalake",
        "LATEST_DATE": "2024-01-22",
    })

    assert github_env.read_text(encoding="utf-8") == (
        "MODEL_VERSION=3\n"
        "STORAGE_TYPE=datalake\n"
        "LATEST_DATE=2024-01-22\n"
    )


def test_write_environment_variables_appends_to_existing_file(
    monkeypatch,
    tmp_path,
):
    # Purpose: Check that existing GitHub Actions environment variables are
    # preserved when new values are written.
    github_env = tmp_path / "github_env"
    github_env.write_text(
        "EXISTING_VARIABLE=existing-value\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_ENV", str(github_env))

    write_environment_variables({
        "NEW_VARIABLE": "new-value",
    })

    assert github_env.read_text(encoding="utf-8") == (
        "EXISTING_VARIABLE=existing-value\n"
        "NEW_VARIABLE=new-value\n"
    )


def test_write_environment_variables_creates_file_when_it_does_not_exist(
    monkeypatch,
    tmp_path,
):
    # Purpose: Check that the GitHub environment file is created when its
    # parent directory exists but the file itself does not.
    github_env = tmp_path / "new_github_env"

    monkeypatch.setenv("GITHUB_ENV", str(github_env))

    write_environment_variables({
        "RESOURCE_GROUP": "test-rg",
    })

    assert github_env.exists()
    assert github_env.read_text(encoding="utf-8") == (
        "RESOURCE_GROUP=test-rg\n"
    )


def test_write_environment_variables_with_empty_values_writes_nothing(
    monkeypatch,
    tmp_path,
):
    # Purpose: Check that supplying no environment variables leaves an
    # existing GitHub environment file unchanged.
    github_env = tmp_path / "github_env"
    github_env.write_text(
        "EXISTING=value\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GITHUB_ENV", str(github_env))

    write_environment_variables({})

    assert github_env.read_text(encoding="utf-8") == (
        "EXISTING=value\n"
    )


def test_write_environment_variables_raises_when_github_env_is_unset(
    monkeypatch,
):
    # Purpose: Check that execution outside GitHub Actions produces a clear
    # error instead of attempting to write to an undefined path.
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    with pytest.raises(
        RuntimeError,
        match=(
            r"GITHUB_ENV is not set\. "
            r"This script must run inside GitHub Actions\."
        ),
    ):
        write_environment_variables({
            "MODEL_VERSION": "3",
        })


def test_write_environment_variables_raises_when_github_env_is_empty(
    monkeypatch,
):
    # Purpose: Check that an empty GITHUB_ENV value is treated as missing.
    monkeypatch.setenv("GITHUB_ENV", "")

    with pytest.raises(
        RuntimeError,
        match=r"GITHUB_ENV is not set",
    ):
        write_environment_variables({
            "MODEL_VERSION": "3",
        })