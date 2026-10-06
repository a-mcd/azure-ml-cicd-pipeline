"""Integration tests for the get_latest_date_csv script."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_get_latest_date_csv(
    csv_path: Path,
    *,
    github_env: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run get_latest_date_csv.py as a real CLI process."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    command = [
        sys.executable,
        "-m",
        "walmart_ml.pipeline.get_latest_date_csv",
        "--file",
        str(csv_path),
    ]

    if github_env is not None:
        environment["GITHUB_ENV"] = str(github_env)
        command.append("--github_env")

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_finds_latest_date_and_writes_github_environment(
    tmp_path,
):
    """Process a real CSV and write the calculated dates."""
    csv_path = tmp_path / "walmart-sales.csv"
    github_env = tmp_path / "github-env.txt"

    csv_path.write_text(
        "Store,Date,Weekly_Sales\n"
        "1,19-10-2012,100000.00\n"
        "2,02-11-2012,200000.00\n"
        "3,,150000.00\n"
        "4,26-10-2012,175000.00\n",
        encoding="utf-8",
    )

    result = run_get_latest_date_csv(
        csv_path,
        github_env=github_env,
    )

    assert result.returncode == 0
    assert result.stderr == ""

    assert "Latest Date: 02-11-2012" in result.stdout
    assert (
        "Materialisation End Date: 09-11-2012"
        in result.stdout
    )

    assert github_env.is_file()

    environment_output = github_env.read_text(
        encoding="utf-8"
    )

    assert "LATEST_DATE=02-11-2012" in environment_output
    assert (
        "MATERIALISATION_END_DATE=09-11-2012"
        in environment_output
    )


def test_supports_utf8_bom_csv_files(tmp_path):
    """Read CSV files containing a UTF-8 byte-order mark."""
    csv_path = tmp_path / "walmart-sales-bom.csv"

    csv_path.write_text(
        "Store,Date,Weekly_Sales\n"
        "1,26-10-2012,100000.00\n"
        "2,02-11-2012,200000.00\n",
        encoding="utf-8-sig",
    )

    result = run_get_latest_date_csv(csv_path)

    assert result.returncode == 0
    assert "Latest Date: 02-11-2012" in result.stdout
    assert (
        "Materialisation End Date: 09-11-2012"
        in result.stdout
    )


def test_fails_when_date_column_is_missing(tmp_path):
    """Return a non-zero status when Date is absent."""
    csv_path = tmp_path / "missing-date-column.csv"

    csv_path.write_text(
        "Store,Weekly_Sales\n"
        "1,100000.00\n",
        encoding="utf-8",
    )

    result = run_get_latest_date_csv(csv_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "ERROR:" in result.stderr
    assert "Column 'Date' was not found" in result.stderr
    assert "Available columns: Store, Weekly_Sales" in result.stderr


def test_reports_invalid_date_and_csv_row(tmp_path):
    """Return a useful error for malformed dates."""
    csv_path = tmp_path / "invalid-date.csv"

    csv_path.write_text(
        "Store,Date,Weekly_Sales\n"
        "1,26-10-2012,100000.00\n"
        "2,2012-11-02,200000.00\n",
        encoding="utf-8",
    )

    result = run_get_latest_date_csv(csv_path)

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
    assert (
        "Date '2012-11-02' does not match any "
        "accepted format"
        in result.stderr
    )
    assert "at CSV row 3" in result.stderr


def test_fails_when_all_dates_are_empty(tmp_path):
    """Reject a CSV with no usable dates."""
    csv_path = tmp_path / "empty-dates.csv"

    csv_path.write_text(
        "Store,Date,Weekly_Sales\n"
        "1,,100000.00\n"
        "2,   ,200000.00\n",
        encoding="utf-8",
    )

    result = run_get_latest_date_csv(csv_path)

    assert result.returncode == 1
    assert (
        "No non-empty dates were found in column 'Date'"
        in result.stderr
    )


def test_fails_when_csv_file_does_not_exist(tmp_path):
    """Return a non-zero status for a missing input file."""
    csv_path = tmp_path / "missing.csv"

    result = run_get_latest_date_csv(csv_path)

    assert result.returncode == 1
    assert "CSV file not found:" in result.stderr
    assert str(csv_path) in result.stderr