"""Unit tests for the batch prediction validation script."""

import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from walmart_ml.pipeline import validate_predictions as validation


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Return valid prediction data before type conversion."""
    return pd.DataFrame(
        {
            "Store": ["1", "2"],
            "Feature_Date": ["02-11-2012", "02-11-2012"],
            "Prediction_Date": ["09-11-2012", "09-11-2012"],
            "Predicted_Weekly_Sales": ["100000.50", "200000.75"],
        }
    )


@pytest.fixture
def converted_dataframe(valid_dataframe) -> pd.DataFrame:
    """Return valid prediction data with converted column types."""
    dataframe = valid_dataframe.copy()
    validation.validate_stores(dataframe)
    validation.validate_dates(dataframe)
    validation.validate_predictions(dataframe)
    return dataframe


def assert_validation_failure(
    function,
    expected_message: str,
    capsys,
) -> None:
    """Assert that a validation function exits with an error message."""
    with pytest.raises(SystemExit) as error:
        function()

    assert error.value.code == 1
    assert expected_message in capsys.readouterr().err


def test_fail_prints_message_and_exits(capsys):
    assert_validation_failure(
        lambda: validation.fail("Something went wrong."),
        "Validation failed: Something went wrong.",
        capsys,
    )


def test_parse_arguments_returns_file_path(mocker, tmp_path):
    predictions_file = tmp_path / "predictions.csv"

    mocker.patch.object(
        sys,
        "argv",
        ["validate_predictions.py", str(predictions_file)],
    )

    arguments = validation.parse_arguments()

    assert isinstance(arguments, argparse.Namespace)
    assert arguments.file == predictions_file


def test_read_predictions_reads_headerless_file(tmp_path):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text(
        "1 02-11-2012 09-11-2012 100000.50\n"
        "2 02-11-2012 09-11-2012 200000.75\n",
        encoding="utf-8",
    )

    dataframe = validation.read_predictions(predictions_file)

    assert dataframe.columns.tolist() == validation.PREDICTION_COLUMNS
    assert len(dataframe) == 2
    assert dataframe.iloc[0]["Store"] == 1
    assert dataframe.iloc[0]["Feature_Date"] == "02-11-2012"


def test_read_predictions_fails_when_file_does_not_exist(
    tmp_path,
    capsys,
):
    missing_file = tmp_path / "missing.csv"

    assert_validation_failure(
        lambda: validation.read_predictions(missing_file),
        f"File not found: {missing_file}",
        capsys,
    )


@pytest.mark.parametrize(
    "exception",
    [
        OSError("read failure"),
        pd.errors.EmptyDataError("empty file"),
        pd.errors.ParserError("invalid file"),
    ],
)
def test_read_predictions_handles_read_errors(
    mocker,
    tmp_path,
    capsys,
    exception,
):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.touch()

    mocker.patch.object(
        validation.pd,
        "read_csv",
        side_effect=exception,
    )

    assert_validation_failure(
        lambda: validation.read_predictions(predictions_file),
        f"Could not read {predictions_file}",
        capsys,
    )


def test_read_predictions_fails_when_dataframe_is_empty(
    mocker,
    tmp_path,
    capsys,
):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.touch()

    mocker.patch.object(
        validation.pd,
        "read_csv",
        return_value=pd.DataFrame(),
    )

    assert_validation_failure(
        lambda: validation.read_predictions(predictions_file),
        "The predictions file contains no rows.",
        capsys,
    )


def test_read_predictions_fails_with_wrong_number_of_columns(
    tmp_path,
    capsys,
):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text(
        "1 02-11-2012 09-11-2012\n",
        encoding="utf-8",
    )

    assert_validation_failure(
        lambda: validation.read_predictions(predictions_file),
        "Expected 4 columns, but found 3.",
        capsys,
    )


def test_validate_no_header_accepts_headerless_data(valid_dataframe):
    validation.validate_no_header(valid_dataframe)


def test_validate_no_header_rejects_header_row(capsys):
    dataframe = pd.DataFrame(
        [validation.PREDICTION_COLUMNS],
        columns=validation.PREDICTION_COLUMNS,
    )

    assert_validation_failure(
        lambda: validation.validate_no_header(dataframe),
        "The file contains a header row",
        capsys,
    )


def test_validate_stores_converts_stores_to_integers(valid_dataframe):
    validation.validate_stores(valid_dataframe)

    assert valid_dataframe["Store"].tolist() == [1, 2]
    assert valid_dataframe["Store"].dtype == np.dtype("int64")


@pytest.mark.parametrize("invalid_store", ["invalid", None, np.nan])
def test_validate_stores_rejects_non_numeric_values(
    valid_dataframe,
    capsys,
    invalid_store,
):
    valid_dataframe.loc[0, "Store"] = invalid_store

    assert_validation_failure(
        lambda: validation.validate_stores(valid_dataframe),
        "Store contains non-numeric or null values",
        capsys,
    )


def test_validate_stores_rejects_non_integer_values(
    valid_dataframe,
    capsys,
):
    valid_dataframe.loc[0, "Store"] = 1.5

    assert_validation_failure(
        lambda: validation.validate_stores(valid_dataframe),
        "Store contains non-integer values",
        capsys,
    )


@pytest.mark.parametrize("invalid_store", [0, 46, -1])
def test_validate_stores_rejects_values_outside_range(
    valid_dataframe,
    capsys,
    invalid_store,
):
    valid_dataframe.loc[0, "Store"] = invalid_store

    assert_validation_failure(
        lambda: validation.validate_stores(valid_dataframe),
        "Store must be between 1 and 45",
        capsys,
    )


def test_validate_dates_converts_date_columns(valid_dataframe):
    validation.validate_dates(valid_dataframe)

    assert pd.api.types.is_datetime64_any_dtype(
        valid_dataframe["Feature_Date"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        valid_dataframe["Prediction_Date"]
    )

    assert valid_dataframe.loc[0, "Feature_Date"] == pd.Timestamp(
        "2012-11-02"
    )
    assert valid_dataframe.loc[0, "Prediction_Date"] == pd.Timestamp(
        "2012-11-09"
    )


@pytest.mark.parametrize(
    ("column", "invalid_date"),
    [
        ("Feature_Date", "2012-11-02"),
        ("Prediction_Date", "not-a-date"),
        ("Prediction_Date", "32-11-2012"),
    ],
)
def test_validate_dates_rejects_invalid_dates(
    valid_dataframe,
    capsys,
    column,
    invalid_date,
):
    valid_dataframe.loc[0, column] = invalid_date

    assert_validation_failure(
        lambda: validation.validate_dates(valid_dataframe),
        f"{column} contains an invalid date",
        capsys,
    )


def test_validate_prediction_horizon_accepts_seven_days(
    converted_dataframe,
):
    validation.validate_prediction_horizon(converted_dataframe)


def test_validate_prediction_horizon_rejects_wrong_horizon(
    converted_dataframe,
    capsys,
):
    converted_dataframe.loc[0, "Prediction_Date"] = pd.Timestamp(
        "2012-11-10"
    )

    assert_validation_failure(
        lambda: validation.validate_prediction_horizon(
            converted_dataframe
        ),
        "Prediction_Date is not seven days after Feature_Date",
        capsys,
    )


def test_validate_predictions_converts_values_to_numeric(
    valid_dataframe,
):
    validation.validate_predictions(valid_dataframe)

    assert valid_dataframe["Predicted_Weekly_Sales"].tolist() == [
        100000.50,
        200000.75,
    ]
    assert pd.api.types.is_numeric_dtype(
        valid_dataframe["Predicted_Weekly_Sales"]
    )


@pytest.mark.parametrize(
    "invalid_prediction",
    [
        "invalid",
        None,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_predictions_rejects_invalid_values(
    valid_dataframe,
    capsys,
    invalid_prediction,
):
    valid_dataframe.loc[0, "Predicted_Weekly_Sales"] = (
        invalid_prediction
    )

    assert_validation_failure(
        lambda: validation.validate_predictions(valid_dataframe),
        (
            "Predicted_Weekly_Sales contains null, non-numeric, "
            "or non-finite values"
        ),
        capsys,
    )


def test_validate_predictions_accepts_negative_sales(
    valid_dataframe,
):
    valid_dataframe.loc[0, "Predicted_Weekly_Sales"] = -100.0

    validation.validate_predictions(valid_dataframe)

    assert valid_dataframe.loc[0, "Predicted_Weekly_Sales"] == -100.0


def test_validate_duplicates_accepts_unique_rows(
    converted_dataframe,
):
    validation.validate_duplicates(converted_dataframe)


def test_validate_duplicates_rejects_duplicate_store_and_date(
    converted_dataframe,
    capsys,
):
    duplicate = converted_dataframe.iloc[[0]].copy()
    dataframe = pd.concat(
        [converted_dataframe, duplicate],
        ignore_index=True,
    )

    assert_validation_failure(
        lambda: validation.validate_duplicates(dataframe),
        "Duplicate Store/Feature_Date predictions found",
        capsys,
    )


def test_validate_duplicates_allows_same_store_on_different_dates(
    converted_dataframe,
):
    additional_row = converted_dataframe.iloc[[0]].copy()
    additional_row["Feature_Date"] = pd.Timestamp("2012-11-09")
    additional_row["Prediction_Date"] = pd.Timestamp("2012-11-16")

    dataframe = pd.concat(
        [converted_dataframe, additional_row],
        ignore_index=True,
    )

    validation.validate_duplicates(dataframe)


def test_main_validates_file_successfully(
    mocker,
    tmp_path,
    capsys,
):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text(
        "1 02-11-2012 09-11-2012 100000.50\n"
        "2 02-11-2012 09-11-2012 200000.75\n",
        encoding="utf-8",
    )

    mocker.patch.object(
        sys,
        "argv",
        ["validate_predictions.py", str(predictions_file)],
    )

    validation.main()

    captured = capsys.readouterr()

    assert (
        f"All 2 prediction rows validated successfully "
        f"in {predictions_file}."
        in captured.out
    )


def test_main_stops_when_validation_fails(
    mocker,
    tmp_path,
    capsys,
):
    predictions_file = tmp_path / "predictions.csv"
    predictions_file.write_text(
        "46 02-11-2012 09-11-2012 100000.50\n",
        encoding="utf-8",
    )

    mocker.patch.object(
        sys,
        "argv",
        ["validate_predictions.py", str(predictions_file)],
    )

    with pytest.raises(SystemExit) as error:
        validation.main()

    assert error.value.code == 1
    assert "Store must be between 1 and 45" in capsys.readouterr().err