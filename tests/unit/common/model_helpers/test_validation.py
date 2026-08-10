import pandas as pd
import pytest
from  src.common.model_helpers.validation import validate_missing_values


def test_validate_missing_values_passes_when_required_columns_have_no_missing_values():
    # Purpose: Check that validation passes when all required columns exist
    # and contain no missing values.
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Weekly_Sales": [1000.0, 2000.0, 3000.0],
        "Temperature": [20.5, 21.0, 19.8],
    })

    validate_missing_values(
        df=df,
        set_name="training",
        required_cols=["Store", "Weekly_Sales", "Temperature"],
    )


def test_validate_missing_values_raises_error_when_required_column_is_missing():
    # Purpose: Check that validation raises a ValueError when a required
    # column is missing.
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Weekly_Sales": [1000.0, 2000.0, 3000.0],
    })

    with pytest.raises(ValueError) as exc_info:
        validate_missing_values(
            df=df,
            set_name="training",
            required_cols=["Store", "Weekly_Sales", "Temperature"],
        )

    error_message = str(exc_info.value)

    assert "Dataset training is missing required columns" in error_message
    assert "Temperature" in error_message


def test_validate_missing_values_raises_error_when_required_column_contains_nan():
    # Purpose: Check that validation raises a ValueError when a required
    # column contains missing values.
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Weekly_Sales": [1000.0, None, None],
        "Temperature": [20.5, 21.0, 19.8],
    })

    with pytest.raises(ValueError) as exc_info:
        validate_missing_values(
            df=df,
            set_name="training",
            required_cols=["Store", "Weekly_Sales", "Temperature"],
        )

    error_message = str(exc_info.value)

    assert "Missing values found in dataset" in error_message
    assert "Weekly_Sales" in error_message
    assert "2" in error_message


def test_validate_missing_values_only_checks_required_columns():
    # Purpose: Check that missing values in optional columns are ignored
    # when those columns are not required.
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Weekly_Sales": [1000.0, 2000.0, 3000.0],
        "Optional_Column": [None, None, None],
    })

    validate_missing_values(
        df=df,
        set_name="training",
        required_cols=["Store", "Weekly_Sales"],
    )