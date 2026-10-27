"""Champion-challenger comparison for the staging workflow.

Compares the challenger model's held-out-week forecast with the champion
endpoint's predictions for the same stores and dates. Metrics are compared in
order (WAPE, then RMSE, absolute bias and MAPE), and a difference within a
metric's tolerance counts as a tie. The champion wins every tie.

MAE and R2 are not used to rank the models. Both models are scored on the
same week, so MAE always ranks them the same way as WAPE, and R2 the same way
as RMSE.

The metrics and winner are printed, a Markdown summary is written to
``challenger_champion_summary.txt``, and ``winner=<champion|challenger>`` is
appended to the GitHub Actions output file.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from walmart_ml.logging_config import configure_logging
from walmart_ml.training import config

LOGGER = logging.getLogger(__name__)


# Differences within these tolerances count as ties.
WAPE_TOLERANCE = 0.1  # percentage points
RMSE_RELATIVE_TOLERANCE = 0.01  # fraction of the champion's RMSE
BIAS_TOLERANCE = 1.0  # percentage points of total actual sales
MAPE_TOLERANCE = 0.1  # percentage points

DATE_FORMAT = "%d-%m-%Y"


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        argparse.Namespace: Paths to the challenger CSV, the champion JSON and
        the GitHub Actions output file.
    """
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
    """Check that an input file exists and is not empty.

    Args:
        path: File to check.
        description: Name of the file, used in error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")


def load_challenger_predictions(csv_path: Path) -> pd.DataFrame:
    """Load the challenger forecast written by the training job.

    Args:
        csv_path: Path to ``challenger_forecast_by_store.csv``.

    Returns:
        pd.DataFrame: Store, Feature_Date, Prediction_Date, Actual_Next_Week_Sales
        and Challenger_Predicted_Weekly_Sales, with parsed dates.

    Raises:
        ValueError: If required columns are missing or values cannot be parsed.
    """
    predictions = pd.read_csv(csv_path)

    required_columns = set(config.CHALLENGER_FORECAST_COLUMNS)

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
    """Load the champion endpoint's response.

    Args:
        json_path: Path to the endpoint's JSON response.

    Returns:
        pd.DataFrame: Store, Feature_Date, Prediction_Date and
        Champion_Predicted_Weekly_Sales, with parsed dates.

    Raises:
        ValueError: If the response is not in the expected shape, columns are
            missing, or values cannot be parsed.
    """
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
    """Join the challenger and champion predictions row for row.

    Args:
        challenger: Output of :func:`load_challenger_predictions`.
        champion: Output of :func:`load_champion_predictions`.

    Returns:
        pd.DataFrame: One row per Store, Feature_Date and Prediction_Date with
        both models' predictions.

    Raises:
        ValueError: If any row in either input has no match in the other.
    """
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
    """Calculate WAPE, RMSE, MAE, MAPE and bias for one model.

    Args:
        actual: Actual next-week sales.
        predicted: Predicted next-week sales.

    Returns:
        dict[str, float]: Metric values keyed ``wape``, ``rmse``, ``mae``,
        ``mape``, ``bias`` and ``bias_pct``. WAPE and MAPE are percentages.
        ``bias`` is the mean of predicted minus actual sales, so a positive
        value means over-forecasting. ``bias_pct`` is the same total as a
        percentage of total actual sales.

    Raises:
        ValueError: If values are missing, or zero sales make WAPE or MAPE
            undefined.
    """
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
        "bias": float(
            (-errors).mean()
        ),
        "bias_pct": float(
            (-errors).sum() / actual_total * 100
        ),
    }


def select_winner(
    challenger: dict[str, float],
    champion: dict[str, float],
) -> tuple[str, str]:

    """Pick the better model by comparing metrics in priority order.

    The order is WAPE, RMSE, absolute bias and MAPE. Lower is better for
    each. A difference within a metric's tolerance counts as a tie and moves
    on to the next metric, and the champion wins if every metric ties.

    Args:
        challenger: Challenger metrics from :func:`calculate_metrics`.
        champion: Champion metrics from :func:`calculate_metrics`.

    Returns:
        tuple[str, str]: ``"challenger"`` or ``"champion"``, and the reason.
    """
    comparisons = (
        (
            challenger["wape"],
            champion["wape"],
            WAPE_TOLERANCE,
            "lower WAPE",
        ),
        (
            challenger["rmse"],
            champion["rmse"],
            champion["rmse"] * RMSE_RELATIVE_TOLERANCE,
            "lower RMSE after WAPE tie",
        ),
        (
            abs(challenger["bias_pct"]),
            abs(champion["bias_pct"]),
            BIAS_TOLERANCE,
            "lower absolute bias after WAPE and RMSE tie",
        ),
        (
            challenger["mape"],
            champion["mape"],
            MAPE_TOLERANCE,
            "lower MAPE after WAPE, RMSE and bias tie",
        ),
    )

    winner = "champion"
    reason = "metrics effectively tied"

    for challenger_value, champion_value, tolerance, comparison_reason in comparisons:
        difference = champion_value - challenger_value

        if abs(difference) > tolerance:
            winner = "challenger" if difference > 0 else "champion"
            reason = comparison_reason
            break

    return winner, reason


def print_metrics(
    model_name: str,
    metrics: dict[str, float],
) -> None:
    """Print one model's metrics.

    Args:
        model_name: Label printed before each metric.
        metrics: Metrics from :func:`calculate_metrics`.
    """
    print(f"{model_name} WAPE: {metrics['wape']:.4f}%")
    print(f"{model_name} RMSE: {metrics['rmse']:.2f}")
    print(f"{model_name} MAE:  {metrics['mae']:.2f}")
    print(f"{model_name} MAPE: {metrics['mape']:.4f}%")
    print(f"{model_name} Bias: {metrics['bias']:.2f} ({metrics['bias_pct']:.4f}%)")


def write_summary_file(
    path: Path,
    winner: str,
    reason: str,
    challenger: dict[str, float],
    champion: dict[str, float],
) -> None:
    """Write a Markdown summary table of the comparison.

    Args:
        path: File to write. Its parent folder is created if needed.
        winner: Winning model.
        reason: Why that model won.
        challenger: Challenger metrics.
        champion: Champion metrics.
    """
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
        f"| Bias | {challenger['bias']:.2f} ({challenger['bias_pct']:.4f}%) | "
        f"{champion['bias']:.2f} ({champion['bias_pct']:.4f}%) |\n"
        "\nModels are ranked by WAPE, then RMSE, absolute bias and MAPE. "
        "The champion wins ties.\n"
    )

    path.write_text(summary, encoding="utf-8")

    LOGGER.info("Summary written to: %s", path)


def write_github_output(
    path: Path,
    winner: str,
) -> None:

    #FOR TESTING PURPOSES
    #winner="challenger"
    """Append the winner to the GitHub Actions step output file.

    Args:
        path: Path from ``$GITHUB_OUTPUT``.
        winner: Winning model.
    """
    with path.open("a", encoding="utf-8") as output:
        output.write(f"winner={winner}\n")


def main() -> None:
    """Compare the two models and record the winner."""
    configure_logging()
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
