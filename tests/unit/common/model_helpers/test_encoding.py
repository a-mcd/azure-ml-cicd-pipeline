import pandas as pd

from walmart_ml.common.model_helpers.encoding import (
    get_feature_columns,
    prepare_encodings,
)


def test_prepare_encodings_creates_store_one_hot_columns():
    # Purpose: Check that prepare_encodings creates one-hot columns for each store.
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "Holiday": ["N/A", "Christmas", "Thanksgiving"],
        "Weekly_Sales": [1000.0, 2000.0, 3000.0],
    })

    result = prepare_encodings(df)

    assert "S_1" in result.columns
    assert "S_2" in result.columns
    assert "S_3" in result.columns

    assert result.loc[0, "S_1"]
    assert result.loc[1, "S_2"]
    assert result.loc[2, "S_3"]


def test_prepare_encodings_creates_known_holiday_one_hot_columns():
    # Purpose: Check that prepare_encodings creates all expected holiday one-hot columns.
    df = pd.DataFrame({
        "Store": [1, 2],
        "Holiday": ["Christmas", "Labor_Day"],
        "Weekly_Sales": [1000.0, 2000.0],
    })

    result = prepare_encodings(df)

    expected_holiday_columns = [
        "Holiday_Christmas",
        "Holiday_Thanksgiving",
        "Holiday_Labor_Day",
        "Holiday_Super_Bowl",
    ]

    for col in expected_holiday_columns:
        assert col in result.columns
    
    assert "Holiday_N/A" not in result.columns


def test_prepare_encodings_replaces_missing_holiday_with_na_category():
    # Purpose: Check that missing Holiday values are encoded as the Holiday_N/A category.
    df = pd.DataFrame({
        "Store": [1, 2],
        "Holiday": [None, "Christmas"],
        "Weekly_Sales": [1000.0, 2000.0],
    })

    result = prepare_encodings(df)

    assert not result.loc[0, "Holiday_Christmas"]
    assert result.loc[1, "Holiday_Christmas"]


def test_prepare_encodings_removes_original_store_and_holiday_columns():
    # Purpose: Check that original Store and Holiday columns are removed after encoding.
    df = pd.DataFrame({
        "Store": [1],
        "Holiday": ["Christmas"],
        "Weekly_Sales": [1000.0],
    })

    result = prepare_encodings(df)

    assert "Store" not in result.columns
    assert "Holiday" not in result.columns


def test_prepare_encodings_keeps_non_encoded_columns():
    # Purpose: Check that non-encoded columns are preserved after encoding.
    df = pd.DataFrame({
        "Store": [1],
        "Holiday": ["Christmas"],
        "Weekly_Sales": [1000.0],
        "Temperature": [22.5],
    })

    result = prepare_encodings(df)

    assert "Weekly_Sales" in result.columns
    assert "Temperature" in result.columns
    assert result.loc[0, "Weekly_Sales"] == 1000.0
    assert result.loc[0, "Temperature"] == 22.5


def test_prepare_encodings_does_not_modify_original_dataframe():
    # Purpose: Check that prepare_encodings does not mutate the original input DataFrame.
    df = pd.DataFrame({
        "Store": [1],
        "Holiday": [None],
        "Weekly_Sales": [1000.0],
    })

    original_df = df.copy(deep=True)

    prepare_encodings(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_prepare_encodings_handles_unknown_holiday_as_all_false_holiday_columns():
    # Purpose: Check that unknown holidays do not match any known holiday one-hot category.
    df = pd.DataFrame({
        "Store": [1],
        "Holiday": ["Unknown Holiday"],
        "Weekly_Sales": [1000.0],
    })

    result = prepare_encodings(df)

    holiday_cols = [col for col in result.columns if col.startswith("Holiday_")]

    assert holiday_cols
    assert not result.loc[0, holiday_cols].any()


def test_get_feature_columns_returns_base_features_first():
    # Purpose: Check that get_feature_columns returns base numeric/time-series features first.
    train_enc = pd.DataFrame({
        "Temperature": [20.0],
        "Fuel_Price": [3.5],
        "CPI": [220.0],
        "Unemployment": [7.5],
        "lag_1": [100.0],
        "lag_2": [90.0],
        "lag_52": [80.0],
        "rollmean_3": [95.0],
        "rollmean_6": [92.0],
        "rollmean_12": [91.0],
        "month": [1],
        "week": [2],
        "quarter": [1],
        "year": [2024],
        "S_1": [True],
        "Holiday_Christmas": [False],
    })

    result = get_feature_columns(train_enc)

    expected_base_features = [
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "lag_1", "lag_2", "lag_52",
        "rollmean_3", "rollmean_6", "rollmean_12",
        "month", "week", "quarter", "year",
    ]

    assert result[:len(expected_base_features)] == expected_base_features


def test_get_feature_columns_includes_store_and_holiday_onehot_columns():
    # Purpose: Check that store and holiday one-hot columns are included as model features.
    train_enc = pd.DataFrame({
        "Temperature": [20.0],
        "S_1": [True],
        "S_2": [False],
        "Holiday_N/A": [True],
        "Holiday_Christmas": [False],
        "Weekly_Sales": [1000.0],
    })

    result = get_feature_columns(train_enc)

    assert "S_1" in result
    assert "S_2" in result
    assert "Holiday_N/A" in result
    assert "Holiday_Christmas" in result


def test_get_feature_columns_excludes_non_feature_columns():
    # Purpose: Check that target and metadata columns are excluded from model features.
    train_enc = pd.DataFrame({
        "Temperature": [20.0],
        "Weekly_Sales": [1000.0],
        "Date": [pd.Timestamp("2024-01-01")],
        "S_1": [True],
        "Holiday_N/A": [True],
    })

    result = get_feature_columns(train_enc)

    assert "Weekly_Sales" not in result
    assert "Date" not in result


def test_get_feature_columns_returns_base_features_even_if_missing_from_dataframe():
    # Purpose: Check that expected base features are returned even when missing from the encoded DataFrame.
    train_enc = pd.DataFrame({
        "S_1": [True],
        "Holiday_N/A": [True],
    })

    result = get_feature_columns(train_enc)

    assert "Temperature" in result
    assert "Fuel_Price" in result
    assert "Unemployment" in result
    assert "S_1" in result
    assert "Holiday_N/A" in result