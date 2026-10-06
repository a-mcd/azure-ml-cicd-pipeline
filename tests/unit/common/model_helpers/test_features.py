import pandas as pd

from walmart_ml.common.model_helpers.features import (
    align_to_feature_columns,
    build_features,
    prepare_calendar_features,
    prepare_holiday_features,
    prepare_time_series_features,
)


def test_prepare_calendar_features_adds_year_month_week_and_quarter():
    # Purpose: Check that calendar features are created from the Date column.
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-09-15"]),
    })

    result = prepare_calendar_features(df)

    assert result.loc[0, "year"] == 2024
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "week"] == 1
    assert result.loc[0, "quarter"] == 1

    assert result.loc[1, "year"] == 2024
    assert result.loc[1, "month"] == 9
    assert result.loc[1, "quarter"] == 3


def test_prepare_holiday_features_sets_na_when_no_holiday_flag():
    # Purpose: Check that non-holiday rows are labelled as N/A and Holiday_Flag is removed.
    df = pd.DataFrame({
        "month": [1, 3, 5],
        "Holiday_Flag": [0, 0, 0],
    })

    result = prepare_holiday_features(df)

    assert list(result["Holiday"]) == ["N/A", "N/A", "N/A"]
    assert "Holiday_Flag" not in result.columns


def test_prepare_holiday_features_maps_known_holiday_months():
    # Purpose: Check that known holiday months are mapped to the correct holiday names.
    df = pd.DataFrame({
        "month": [11, 12, 2, 9],
        "Holiday_Flag": [1, 1, 1, 1],
    })

    result = prepare_holiday_features(df)

    assert list(result["Holiday"]) == [
        "Thanksgiving",
        "Christmas",
        "Super_Bowl",
        "Labor_Day",
    ]


def test_prepare_holiday_features_handles_missing_holiday_flag_column():
    # Purpose: Check that missing Holiday_Flag is handled by setting Holiday to N/A.
    df = pd.DataFrame({
        "month": [11, 12],
    })

    result = prepare_holiday_features(df)

    assert list(result["Holiday"]) == ["N/A", "N/A"]
    assert "Holiday_Flag" not in result.columns


def test_prepare_time_series_features_creates_lag_columns_and_target_without_dropping_rows():
    # Purpose: Check that lag features and next-week target are created without dropping rows.
    dates = pd.date_range("2023-01-01", periods=55, freq="W")

    df = pd.DataFrame({
        "Store": [1] * 55,
        "Date": dates,
        "Weekly_Sales": range(100, 155),
    })

    result = prepare_time_series_features(df, drop_training_rows=False)

    assert "lag_1" in result.columns
    assert "lag_2" in result.columns
    assert "lag_52" in result.columns
    assert "Weekly_Sales_tplus1" in result.columns

    assert pd.isna(result.loc[0, "lag_1"])
    assert result.loc[1, "lag_1"] == 100
    assert result.loc[2, "lag_2"] == 100
    assert result.loc[52, "lag_52"] == 100
    assert result.loc[0, "Weekly_Sales_tplus1"] == 101


def test_prepare_time_series_features_creates_rolling_mean_columns():
    # Purpose: Check that rolling mean features are created using previous sales values.
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-15",
            "2024-01-22",
        ]),
        "Weekly_Sales": [100.0, 200.0, 300.0, 400.0],
    })

    result = prepare_time_series_features(df, drop_training_rows=False)

    assert "rollmean_3" in result.columns
    assert "rollmean_6" in result.columns
    assert "rollmean_12" in result.columns

    assert pd.isna(result.loc[0, "rollmean_3"])
    assert result.loc[1, "rollmean_3"] == 100.0
    assert result.loc[2, "rollmean_3"] == 150.0
    assert result.loc[3, "rollmean_3"] == 200.0


def test_prepare_time_series_features_keeps_store_histories_separate():
    # Purpose: Check that lag features are calculated separately for each store.
    df = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-01",
            "2024-01-08",
        ]),
        "Weekly_Sales": [100.0, 200.0, 1000.0, 2000.0],
    })

    result = prepare_time_series_features(df, drop_training_rows=False)

    store_1_second_row = result[
        (result["Store"] == 1)
        & (result["Date"] == pd.Timestamp("2024-01-08"))
    ]
    store_2_second_row = result[
        (result["Store"] == 2)
        & (result["Date"] == pd.Timestamp("2024-01-08"))
    ]

    assert store_1_second_row.iloc[0]["lag_1"] == 100.0
    assert store_2_second_row.iloc[0]["lag_1"] == 1000.0


def test_prepare_time_series_features_drops_unusable_training_rows():
    # Purpose: Check that rows with missing lag/target values are dropped for training.
    dates = pd.date_range("2023-01-01", periods=55, freq="W")

    df = pd.DataFrame({
        "Store": [1] * 55,
        "Date": dates,
        "Weekly_Sales": range(100, 155),
    })

    result = prepare_time_series_features(df, drop_training_rows=True)

    assert len(result) == 2
    assert result["lag_1"].notna().all()
    assert result["lag_2"].notna().all()
    assert result["lag_52"].notna().all()
    assert result["Weekly_Sales_tplus1"].notna().all()


def test_build_features_adds_calendar_holiday_time_series_features():
    # Purpose: Check that build_features combines calendar, holiday, and time-series feature engineering.
    dates = pd.date_range("2023-01-01", periods=55, freq="W")

    df = pd.DataFrame({
        "Store": ["1"] * 55,
        "Date": dates,
        "Weekly_Sales": range(100, 155),
        "Holiday_Flag": [0] * 55,
    })

    result = build_features(df, drop_training_rows=False)

    expected_columns = [
        "year",
        "month",
        "week",
        "quarter",
        "Holiday",
        "lag_1",
        "lag_2",
        "lag_52",
        "rollmean_3",
        "rollmean_6",
        "rollmean_12",
        "Weekly_Sales_tplus1",
    ]

    for col in expected_columns:
        assert col in result.columns

    assert result["Store"].dtype == "int64"
    assert "Holiday_Flag" not in result.columns


def test_build_features_does_not_modify_original_dataframe():
    # Purpose: Check that build_features does not mutate the original input DataFrame.
    df = pd.DataFrame({
        "Store": ["1", "1", "1"],
        "Date": pd.to_datetime([
            "2024-01-01",
            "2024-01-08",
            "2024-01-15",
        ]),
        "Weekly_Sales": [100.0, 200.0, 300.0],
        "Holiday_Flag": [0, 0, 0],
    })

    original_df = df.copy(deep=True)

    build_features(df, drop_training_rows=False)

    pd.testing.assert_frame_equal(df, original_df)


def test_align_to_feature_columns_adds_missing_columns_with_zero_and_drops_extra_columns():
    # Purpose: Check that feature alignment adds missing columns with zero and removes extra columns.
    df_enc = pd.DataFrame({
        "Temperature": [20.0],
        "Fuel_Price": [3.5],
        "Extra_Column": [999],
    })

    feature_cols = [
        "Temperature",
        "Fuel_Price",
        "CPI",
    ]

    result = align_to_feature_columns(df_enc, feature_cols)

    assert list(result.columns) == feature_cols
    assert result.loc[0, "Temperature"] == 20.0
    assert result.loc[0, "Fuel_Price"] == 3.5
    assert result.loc[0, "CPI"] == 0
    assert "Extra_Column" not in result.columns


def test_align_to_feature_columns_preserves_requested_column_order():
    # Purpose: Check that feature alignment preserves the exact requested column order.
    df_enc = pd.DataFrame({
        "CPI": [220.0],
        "Temperature": [20.0],
        "Fuel_Price": [3.5],
    })

    feature_cols = [
        "Temperature",
        "Fuel_Price",
        "CPI",
    ]

    result = align_to_feature_columns(df_enc, feature_cols)

    assert list(result.columns) == feature_cols