import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WAPE_TOLERANCE = 0.01
RMSE_TOLERANCE = 1.0
MAPE_TOLERANCE = 0.01
MAE_TOLERANCE = 1.0

DATE_FORMAT = "%d-%m-%Y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare challenger and champion predictions."
    )
    parser.add_argument(
        "--challenger-predictions",
        type=Path,
        required=True,
        help="Path to the challenger predictions CSV.",
    )
    parser.add_argument(
        "--champion-predictions",
        type=Path,
        required=True,
        help="Path to the champion predictions JSON.",
    )

    parser.add_argument(
        "--github-output",
        type=Path,
        required=True,
        help="GitHub Actions output file.",
    )

    return parser.parse_args()


def validate_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")


def load_challenger_predictions(csv_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(csv_path)

    required_columns = {
        "Store",
        "Date",
        "Next_Week_Date",
        "Weekly_Sales",
        "Actual_Next_Week_Sales",
        "Predicted_Weekly_Sales",
    }

    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Challenger CSV is missing columns: {sorted(missing_columns)}"
        )

    predictions = (
        predictions
        .drop(columns=["Weekly_Sales"])
        .rename(
            columns={
                "Date": "Feature_Date",
                "Next_Week_Date": "Prediction_Date",
                "Predicted_Weekly_Sales":
                    "Challenger_Predicted_Weekly_Sales",
            }
        )
    )

    predictions["Store"] = pd.to_numeric(
        predictions["Store"],
        errors="raise",
    ).astype(int)

    predictions["Feature_Date"] = pd.to_datetime(
        predictions["Feature_Date"],
        format=DATE_FORMAT,
        errors="raise",
    )

    predictions["Prediction_Date"] = pd.to_datetime(
        predictions["Prediction_Date"],
        format=DATE_FORMAT,
        errors="raise",
    )

    return predictions


def load_champion_predictions(json_path: Path) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as file:
        response = json.load(file)

    if not isinstance(response, dict):
        raise ValueError("Champion response must be a JSON object.")

    predictions_data = response.get("predictions")

    if not isinstance(predictions_data, list):
        raise ValueError(
            "Champion response must contain a 'predictions' list."
        )

    predictions = pd.DataFrame(predictions_data)

    required_columns = {
        "Store",
        "Feature_Date",
        "Prediction_Date",
        "Predicted_Weekly_Sales",
    }

    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Champion predictions are missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions = predictions.rename(
        columns={
            "Predicted_Weekly_Sales":
                "Champion_Predicted_Weekly_Sales",
        }
    )

    predictions["Store"] = pd.to_numeric(
        predictions["Store"],
        errors="raise",
    ).astype(int)

    predictions["Feature_Date"] = pd.to_datetime(
        predictions["Feature_Date"],
        format=DATE_FORMAT,
        errors="raise",
    )

    predictions["Prediction_Date"] = pd.to_datetime(
        predictions["Prediction_Date"],
        format=DATE_FORMAT,
        errors="raise",
    )

    return predictions[
        [
            "Store",
            "Feature_Date",
            "Prediction_Date",
            "Champion_Predicted_Weekly_Sales",
        ]
    ]


def merge_predictions(
    challenger: pd.DataFrame,
    champion: pd.DataFrame,
) -> pd.DataFrame:
    comparison = challenger.merge(
        champion,
        on=["Store", "Feature_Date", "Prediction_Date"],
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(challenger):
        raise ValueError(
            "Not all challenger rows matched champion rows. "
            f"Challenger rows: {len(challenger)}, "
            f"matched rows: {len(comparison)}."
        )

    if len(comparison) != len(champion):
        raise ValueError(
            "Not all champion rows matched challenger rows. "
            f"Champion rows: {len(champion)}, "
            f"matched rows: {len(comparison)}."
        )

    return comparison


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    actual = pd.to_numeric(actual, errors="raise")
    predicted = pd.to_numeric(predicted, errors="raise")

    if actual.isna().any() or predicted.isna().any():
        raise ValueError("Actual or predicted sales contain missing values.")

    errors = actual - predicted
    absolute_errors = errors.abs()

    actual_total = actual.abs().sum()
    if actual_total == 0:
        raise ValueError(
            "Cannot calculate WAPE because total actual sales are zero."
        )

    if (actual == 0).any():
        raise ValueError(
            "Cannot calculate MAPE because actual sales contain zero values."
        )

    return {
        "wape": float(
            absolute_errors.sum() / actual_total * 100
        ),
        "rmse": float(
            np.sqrt(np.mean(errors**2))
        ),
        "mae": float(
            absolute_errors.mean()
        ),
        "mape": float(
            (absolute_errors / actual.abs()).mean() * 100
        ),
    }


def select_winner(
    challenger: dict[str, float],
    champion: dict[str, float],
) -> tuple[str, str]:

    comparisons = (
        ("wape", WAPE_TOLERANCE, "lower WAPE"),
        ("rmse", RMSE_TOLERANCE, "lower RMSE after WAPE tie"),
        (
            "mape",
            MAPE_TOLERANCE,
            "lower MAPE after WAPE and RMSE tie",
        ),
        (
            "mae",
            MAE_TOLERANCE,
            "lower MAE after other metrics tied",
        ),
    )

    winner = "champion"
    reason = "metrics effectively tied"

    for metric, tolerance, comparison_reason in comparisons:
        difference = champion[metric] - challenger[metric]

        if abs(difference) > tolerance:
            winner = "challenger" if difference > 0 else "champion"
            reason = comparison_reason
            break

    return winner, reason


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    print(f"{model_name} WAPE: {metrics['wape']:.4f}%")
    print(f"{model_name} RMSE: {metrics['rmse']:.2f}")
    print(f"{model_name} MAE:  {metrics['mae']:.2f}")
    print(f"{model_name} MAPE: {metrics['mape']:.4f}%")


def write_summary_file(
    path: Path,
    winner: str,
    reason: str,
    challenger: dict[str, float],
    champion: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        "# Champion–Challenger Comparison\n\n"
        f"**Winner:** `{winner}`\n\n"
        f"**Reason:** {reason}\n\n"
        "| Metric | Challenger | Champion |\n"
        "|---|---:|---:|\n"
        f"| WAPE | {challenger['wape']:.4f}% | "
        f"{champion['wape']:.4f}% |\n"
        f"| RMSE | {challenger['rmse']:.2f} | "
        f"{champion['rmse']:.2f} |\n"
        f"| MAE | {challenger['mae']:.2f} | "
        f"{champion['mae']:.2f} |\n"
        f"| MAPE | {challenger['mape']:.4f}% | "
        f"{champion['mape']:.4f}% |\n"
    )

    path.write_text(summary, encoding="utf-8")

    print(f"Summary written to: {path}")


def write_github_output(
    path: Path,
    winner: str,
) -> None:

    #FOR TESTING PURPOSES
    #winner="challenger"
    with path.open("a", encoding="utf-8") as output:
        output.write(f"winner={winner}\n")


def main() -> None:
    args = parse_args()

    validate_file(
        args.challenger_predictions,
        "Challenger predictions",
    )
    validate_file(
        args.champion_predictions,
        "Champion predictions",
    )

    challenger_predictions = load_challenger_predictions(
        args.challenger_predictions
    )
    champion_predictions = load_champion_predictions(
        args.champion_predictions
    )

    comparison = merge_predictions(
        challenger_predictions,
        champion_predictions,
    )

    actual = comparison["Actual_Next_Week_Sales"]

    challenger_metrics = calculate_metrics(
        actual,
        comparison["Challenger_Predicted_Weekly_Sales"],
    )

    champion_metrics = calculate_metrics(
        actual,
        comparison["Champion_Predicted_Weekly_Sales"],
    )

    comparison.to_csv(
        "challenger_champion_comparison.csv",
        index=False,
    )

    winner, reason = select_winner(
        challenger_metrics,
        champion_metrics,
    )

    print()
    print_metrics("Challenger", challenger_metrics)
    print()
    print_metrics("Champion", champion_metrics)
    print()
    print(f"Winner: {winner}")
    print(f"Reason: {reason}")

    write_summary_file(
        Path("challenger_champion_summary.txt"),
        winner,
        reason,
        challenger_metrics,
        champion_metrics,
    )

    write_github_output(
        args.github_output,
        winner,
    )


if __name__ == "__main__":
    main()
