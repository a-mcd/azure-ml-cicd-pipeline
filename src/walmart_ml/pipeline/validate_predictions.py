"""Check the batch endpoint's predictions file before it is published.

The file is the headerless, space-separated output of an Azure ML
``append_row`` batch job, with columns Store, Feature_Date, Prediction_Date and
Predicted_Weekly_Sales. The script prints a success message, or prints the
first problem found to stderr and exits with code 1.

Usage:
    python src/walmart_ml/pipeline/validate_predictions.py predictions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "Store",
    "Feature_Date",
    "Prediction_Date",
    "Predicted_Weekly_Sales",
]

DATE_COLUMNS = ["Feature_Date", "Prediction_Date"]
DATE_FORMAT = "%d-%m-%Y"

def fail(message: str) -> NoReturn:
    """Print a validation error to stderr and exit with code 1.

    Args:
        message: Description of the problem.
    """
    print(f"Validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        argparse.Namespace: Parsed ``file`` path.
    """
    parser = argparse.ArgumentParser(
        description="Validate an Azure ML Walmart batch predictions file."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the headerless predictions CSV file.",
    )
    return parser.parse_args()

def read_predictions(file_path: Path) -> pd.DataFrame:
    """Read the headerless predictions file.

    Exits through :func:`fail` if the file is missing, unreadable, empty, or has
    the wrong number of columns.

    Args:
        file_path: Path to the predictions file.

    Returns:
        pd.DataFrame: Predictions with PREDICTION_COLUMNS as column names.
    """
    if not file_path.is_file():
        fail(f"File not found: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            engine="python",
            header=None,
        )
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        fail(f"Could not read {file_path}: {error}")

    if df.empty:
        fail("The predictions file contains no rows.")

    if len(df.columns) != len(PREDICTION_COLUMNS):
        fail(
            f"Expected {len(PREDICTION_COLUMNS)} columns, "
            f"but found {len(df.columns)}."
        )

    df.columns = PREDICTION_COLUMNS
    return df

def validate_no_header(df: pd.DataFrame) -> None:
    """Reject an accidentally included header row.

    Args:
        df: Predictions from :func:`read_predictions`.
    """
    if df.iloc[0].astype(str).tolist() == PREDICTION_COLUMNS:
        fail(
            "The file contains a header row, but an Azure ML append_row "
            "output was expected to be headerless."
        )

def validate_stores(df: pd.DataFrame) -> None:
    """Check that Store holds whole numbers from 1 to 45, and convert it to int.

    Args:
        df: Predictions. The Store column is converted in place.
    """
    stores = pd.to_numeric(df["Store"], errors="coerce")

    if stores.isna().any():
        invalid = df.loc[stores.isna(), "Store"].head(10).tolist()
        fail(f"Store contains non-numeric or null values: {invalid}")

    non_integer = stores % 1 != 0
    if non_integer.any():
        invalid = stores.loc[non_integer].head(10).tolist()
        fail(f"Store contains non-integer values: {invalid}")

    stores = stores.astype("int64")
    invalid_range = ~stores.between(1, 45)

    if invalid_range.any():
        invalid = stores.loc[invalid_range].head(10).tolist()
        fail(f"Store must be between 1 and 45. Invalid values: {invalid}")

    df["Store"] = stores

def validate_dates(df: pd.DataFrame) -> None:
    """Check that the date columns use DD-MM-YYYY, and convert them to datetime.

    Args:
        df: Predictions. The date columns are converted in place.
    """
    for column in DATE_COLUMNS:
        try:
            df[column] = pd.to_datetime(
                df[column],
                format=DATE_FORMAT,
                errors="raise",
            )
        except (ValueError, TypeError) as error:
            fail(
                f"{column} contains an invalid date. "
                f"Expected DD-MM-YYYY: {error}"
            )

def validate_prediction_horizon(df: pd.DataFrame) -> None:
    """Ensure Prediction_Date is seven days after Feature_Date.

    Args:
        df: Predictions with datetime date columns.
    """
    expected_dates = df["Feature_Date"] + pd.Timedelta(days=7)
    invalid_dates = df["Prediction_Date"] != expected_dates

    if not invalid_dates.any():
        return

    examples = df.loc[
        invalid_dates,
        ["Store", "Feature_Date", "Prediction_Date"],
    ].head(10).copy()

    for column in DATE_COLUMNS:
        examples[column] = examples[column].dt.strftime(DATE_FORMAT)

    fail(
        "Prediction_Date is not seven days after Feature_Date. "
        f"Examples: {examples.to_dict('records')}"
    )

def validate_predictions(df: pd.DataFrame) -> None:
    """Check that predicted sales are finite numbers, and convert them to float.

    Args:
        df: Predictions. Predicted_Weekly_Sales is converted in place.
    """
    predictions = pd.to_numeric(
        df["Predicted_Weekly_Sales"],
        errors="coerce",
    )
    invalid_predictions = predictions.isna() | ~np.isfinite(predictions)

    if invalid_predictions.any():
        invalid = (
            df.loc[invalid_predictions, "Predicted_Weekly_Sales"]
            .head(10)
            .tolist()
        )
        fail(
            "Predicted_Weekly_Sales contains null, non-numeric, or "
            f"non-finite values: {invalid}"
        )

    df["Predicted_Weekly_Sales"] = predictions

def validate_duplicates(df: pd.DataFrame) -> None:
    """Reject duplicate Store and Feature_Date combinations.

    Args:
        df: Predictions with a datetime Feature_Date column.
    """
    duplicates = df.duplicated(
        subset=["Store", "Feature_Date"],
        keep=False,
    )

    if not duplicates.any():
        return

    examples = (
        df.loc[duplicates, ["Store", "Feature_Date"]]
        .head(10)
        .copy()
    )
    examples["Feature_Date"] = examples["Feature_Date"].dt.strftime(
        DATE_FORMAT
    )

    fail(
        "Duplicate Store/Feature_Date predictions found. "
        f"Examples: {examples.to_dict('records')}"
    )

def main() -> None:
    """Validate an Azure ML batch predictions file."""
    file_path = parse_arguments().file
    df = read_predictions(file_path)

    validate_no_header(df)
    validate_stores(df)
    validate_dates(df)
    validate_prediction_horizon(df)
    validate_predictions(df)
    validate_duplicates(df)

    print(
        f"All {len(df)} prediction rows validated successfully "
        f"in {file_path}."
    )

if __name__ == "__main__":
    main()
