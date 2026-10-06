from pathlib import Path
import os
import pandas as pd

def get_outputs_dir() -> Path:
    """
    Get the directory where output files should be saved.

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
    """
    Load the Walmart sales dataset from a CSV file.

    Converts the Date column to a pandas datetime format using day-first
    parsing. If any dates are missing or cannot be parsed, the function
    prints the affected rows and raises an error to stop the script.

    Args:
        path (str): Path to the Walmart CSV file.

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
        print("Found unparsed or missing Date values:")
        print(bad_rows[["Store", "Date"]])
        print("Raw date strings causing issues:")
        print(raw_dates.loc[bad_rows.index])
        raise ValueError(
            f"Invalid or missing Date values found in the Walmart dataset: {len(bad_rows)} rows."
        )

    print("All dates parsed successfully.")

    return df


def chronological_split(df: pd.DataFrame, test_horizon_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split the dataset into training and test sets using chronological order.

    The most recent `test_horizon_weeks` unique dates are used as the test set,
    while all earlier dates are used for training. This avoids random splitting
    and better matches a real forecasting scenario where future weeks are
    predicted from past data.

    Args:
        df (pd.DataFrame): Dataset containing a Date column.
        test_horizon_weeks (int): Number of most recent weeks to use for testing.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]: Training set, test set,
        and the cutoff date used for the split.

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

    print(f"Train end date: {cutoff_date.date()} | Train rows: {len(train):,} | Test rows: {len(test):,}")

    return train, test
