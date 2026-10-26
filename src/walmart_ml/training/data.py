"""Loading the Walmart sales CSV and splitting it by date."""

from pathlib import Path
import logging
import os

import pandas as pd

LOGGER = logging.getLogger(__name__)


def get_outputs_dir() -> Path:
    """Get the directory where output files should be saved.

    If the script is running in Azure ML, use the Azure ML outputs path
    provided by the AZUREML_OUTPUTS_PATH environment variable. If that
    variable is not set, or if the script is running locally, use a local
    folder named 'outputs'.

    The directory is created if it does not already exist.

    Returns:
        Path: The directory path for saving output files.
    """
    if "AZUREML_RUN_ID" in os.environ or "AZUREML_OUTPUTS_PATH" in os.environ:
        outputs_dir = Path(os.getenv("AZUREML_OUTPUTS_PATH", "outputs"))
    else:
        outputs_dir = Path("outputs")

    outputs_dir.mkdir(parents=True, exist_ok=True)

    return outputs_dir


def load_dataset(path: str) -> pd.DataFrame:
    """Load the Walmart sales dataset from a CSV file.

    Converts the Date column to a pandas datetime format using day-first
    parsing. If any dates are missing or cannot be parsed, the function
    logs the affected rows and raises an error to stop the script.

    Args:
        path: Path to the Walmart CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with the Date column converted to datetime.

    Raises:
        ValueError: If any Date values are missing or cannot be parsed.
    """
    df = pd.read_csv(path)
    raw_dates = df["Date"].copy()  # keep original for debugging
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Debug: find any unparsed or missing dates
    bad_rows = df[df["Date"].isna()]
    if not bad_rows.empty:
        LOGGER.error(
            "Found unparsed or missing Date values:\n%s",
            bad_rows[["Store", "Date"]].to_string(),
        )
        LOGGER.error(
            "Raw date strings causing issues:\n%s",
            raw_dates.loc[bad_rows.index].to_string(),
        )
        raise ValueError(
            f"Invalid or missing Date values found in the Walmart dataset: {len(bad_rows)} rows."
        )

    LOGGER.info("All dates parsed successfully.")

    return df


def chronological_split(df: pd.DataFrame, test_horizon_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and test sets using chronological order.

    The most recent ``test_horizon_weeks`` unique dates are used as the test set,
    while all earlier dates are used for training. This avoids random splitting
    and better matches a real forecasting scenario where future weeks are
    predicted from past data.

    Args:
        df: Dataset containing a Date column.
        test_horizon_weeks: Number of most recent weeks to use for testing.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training set and test set.

    Raises:
        ValueError: If there are not enough unique dates to create the split.
    """
    # Use the last `test_horizon_weeks` dates for test
    last_dates = df["Date"].sort_values().unique()
    if len(last_dates) < test_horizon_weeks + 1:
        raise ValueError("Not enough weeks to create the requested test horizon.")
    cutoff_date = last_dates[-test_horizon_weeks - 1]
    train = df[df["Date"] <= cutoff_date].copy()
    test = df[df["Date"] > cutoff_date].copy()

    LOGGER.info(
        "Train end date: %s | Train rows: %s | Test rows: %s",
        cutoff_date.date(),
        f"{len(train):,}",
        f"{len(test):,}",
    )

    return train, test
