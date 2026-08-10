"""Integration tests for the prediction-validation CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "pipeline_scripts"
    / "validate_predictions.py"
)


def run_validation(
    predictions_path: Path,
) -> subprocess.CompletedProcess:
    """Run the prediction validator as a real CLI process."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(predictions_path),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_validates_realistic_batch_predictions(
    tmp_path: Path,
):
    """Validate a headerless prediction file for stores 1–45."""
    predictions_path = tmp_path / "predictions.csv"

    rows = [
        (
            f"{store} "
            "02-11-2012 "
            "09-11-2012 "
            f"{100000.0 + store * 1000.25:.2f}"
        )
        for store in range(1, 46)
    ]

    predictions_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    result = run_validation(predictions_path)

    diagnostics = (
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert result.returncode == 0, diagnostics
    assert result.stderr == ""
    assert (
        f"All 45 prediction rows validated successfully "
        f"in {predictions_path}."
        in result.stdout
    )


def test_rejects_wrong_prediction_horizon(
    tmp_path: Path,
):
    """Reject a prediction that is not seven days ahead."""
    predictions_path = tmp_path / "wrong-horizon.csv"

    predictions_path.write_text(
        "1 02-11-2012 10-11-2012 101000.25\n"
        "2 02-11-2012 09-11-2012 102000.50\n",
        encoding="utf-8",
    )

    result = run_validation(predictions_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Validation failed:" in result.stderr
    assert (
        "Prediction_Date is not seven days after "
        "Feature_Date"
        in result.stderr
    )
    assert "'Store': 1" in result.stderr
    assert "'Feature_Date': '02-11-2012'" in result.stderr
    assert "'Prediction_Date': '10-11-2012'" in result.stderr


def test_rejects_duplicate_store_and_feature_date(
    tmp_path: Path,
):
    """Reject duplicate predictions for the same store/week."""
    predictions_path = tmp_path / "duplicates.csv"

    predictions_path.write_text(
        "1 02-11-2012 09-11-2012 101000.25\n"
        "1 02-11-2012 09-11-2012 102000.50\n",
        encoding="utf-8",
    )

    result = run_validation(predictions_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Validation failed:" in result.stderr
    assert (
        "Duplicate Store/Feature_Date predictions found"
        in result.stderr
    )
    assert "'Store': 1" in result.stderr
    assert "'Feature_Date': '02-11-2012'" in result.stderr


def test_rejects_accidental_header_row(
    tmp_path: Path,
):
    """Reject an Azure append-row output containing a header."""
    predictions_path = tmp_path / "with-header.csv"

    predictions_path.write_text(
        (
            "Store Feature_Date Prediction_Date "
            "Predicted_Weekly_Sales\n"
        )
        + "1 02-11-2012 09-11-2012 101000.25\n",
        encoding="utf-8",
    )

    result = run_validation(predictions_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Validation failed:" in result.stderr
    assert "The file contains a header row" in result.stderr
    assert "expected to be headerless" in result.stderr