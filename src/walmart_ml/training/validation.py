"""Data checks run on the encoded datasets before training."""

import pandas as pd


def validate_missing_values(df: pd.DataFrame, set_name: str, required_cols: list[str]) -> None:
    """Check that the required model input and target columns contain no missing values.

    This validation is run after feature engineering and encoding. Any expected
    missing rows or columns will fail the pipeline.

    Args:
        df: Encoded dataset to validate.
        set_name: Name of the dataset for error messages.
        required_cols: Columns that must not contain missing values.

    Raises:
        ValueError: If any required columns contain missing values in the
        dataset.
    """

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset {set_name} is missing required columns: {missing_cols}")

    missing = df[required_cols].isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        raise ValueError(
            "Missing values found in dataset:\n"
            f"{missing.to_string()}"
        )
