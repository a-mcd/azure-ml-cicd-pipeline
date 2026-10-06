"""Integration test for the champion-challenger CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "walmart_ml"
    / "pipeline_scripts"
    / "select_winner.py"
)


def test_select_winner_cli_creates_expected_outputs(
    tmp_path: Path,
):
    """Run a complete champion-challenger comparison."""
    challenger_path = tmp_path / "challenger.csv"
    champion_path = tmp_path / "champion.json"
    github_output_path = tmp_path / "github-output.txt"

    challenger = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "Date": [
                "02-11-2012",
                "02-11-2012",
                "02-11-2012",
            ],
            "Next_Week_Date": [
                "09-11-2012",
                "09-11-2012",
                "09-11-2012",
            ],
            "Weekly_Sales": [
                95000.0,
                190000.0,
                290000.0,
            ],
            "Actual_Next_Week_Sales": [
                100000.0,
                200000.0,
                300000.0,
            ],
            "Predicted_Weekly_Sales": [
                105000.0,
                195000.0,
                310000.0,
            ],
        }
    )
    challenger.to_csv(challenger_path, index=False)

    champion = {
        "predictions": [
            {
                "Store": 1,
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 120000.0,
            },
            {
                "Store": 2,
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 170000.0,
            },
            {
                "Store": 3,
                "Feature_Date": "02-11-2012",
                "Prediction_Date": "09-11-2012",
                "Predicted_Weekly_Sales": 260000.0,
            },
        ]
    }
    champion_path.write_text(
        json.dumps(champion),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--challenger-predictions",
        str(challenger_path),
        "--champion-predictions",
        str(champion_path),
        "--github-output",
        str(github_output_path),
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    diagnostics = (
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert result.returncode == 0, diagnostics
    assert result.stderr == ""

    assert "Challenger WAPE: 3.3333%" in result.stdout
    assert "Champion WAPE: 15.0000%" in result.stdout
    assert "Winner: challenger" in result.stdout
    assert "Reason: lower WAPE" in result.stdout

    comparison_path = (
        tmp_path / "challenger_champion_comparison.csv"
    )
    summary_path = (
        tmp_path / "challenger_champion_summary.txt"
    )

    assert comparison_path.is_file()
    assert summary_path.is_file()
    assert github_output_path.is_file()

    comparison = pd.read_csv(comparison_path)

    assert comparison.columns.tolist() == [
        "Store",
        "Feature_Date",
        "Prediction_Date",
        "Actual_Next_Week_Sales",
        "Challenger_Predicted_Weekly_Sales",
        "Champion_Predicted_Weekly_Sales",
    ]

    assert len(comparison) == 3
    assert comparison["Store"].tolist() == [1, 2, 3]
    assert comparison["Feature_Date"].tolist() == [
        "2012-11-02",
        "2012-11-02",
        "2012-11-02",
    ]
    assert comparison["Prediction_Date"].tolist() == [
        "2012-11-09",
        "2012-11-09",
        "2012-11-09",
    ]
    assert comparison[
        "Challenger_Predicted_Weekly_Sales"
    ].tolist() == [
        105000.0,
        195000.0,
        310000.0,
    ]
    assert comparison[
        "Champion_Predicted_Weekly_Sales"
    ].tolist() == [
        120000.0,
        170000.0,
        260000.0,
    ]

    summary = summary_path.read_text(encoding="utf-8")

    assert "# Champion–Challenger Comparison" in summary
    assert "**Winner:** `challenger`" in summary
    assert "**Reason:** lower WAPE" in summary
    assert "| WAPE | 3.3333% | 15.0000% |" in summary
    assert "| RMSE | 7071.07 | 31091.26 |" in summary
    assert "| MAE | 6666.67 | 30000.00 |" in summary
    assert "| MAPE | 3.6111% | 16.1111% |" in summary

    assert github_output_path.read_text(
        encoding="utf-8"
    ) == "winner=challenger\n"