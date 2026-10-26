"""Unit tests for building engineered features without the feature store."""

import re
from pathlib import Path

import pandas as pd
import pytest

from walmart_ml.pipeline import build_engineered_features as builder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEATURE_SET_SPEC = (
    PROJECT_ROOT
    / "azureml"
    / "feature_store"
    / "feature_sets"
    / "walmart_sales_features"
    / "spec"
    / "FeatureSetSpec.yaml"
)

WEEKS = 56


def make_raw(stores=(1, 2), weeks=WEEKS) -> pd.DataFrame:
    """Return weekly raw sales rows with a Thanksgiving week in November."""
    dates = pd.date_range("2010-02-05", periods=weeks, freq="7D")
    rows = []
    for store in stores:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "Store": store,
                    "Date": date,
                    "Weekly_Sales": 1000.0 * store + index,
                    "Holiday_Flag": int(date == pd.Timestamp("2010-11-26")),
                    "Temperature": 50.0,
                    "Fuel_Price": 3.0,
                    "CPI": 210.0,
                    "Unemployment": 8.0,
                }
            )
    return pd.DataFrame(rows)


def test_output_columns_match_feature_set_spec():
    # Purpose: the CSV replaces the feature-store export, so its columns
    # must be the feature set's features, in the same order.
    spec = FEATURE_SET_SPEC.read_text(encoding="utf-8").split("index_columns:")[0]
    spec_features = re.findall(r"^\s+- name: (\S+)", spec, re.MULTILINE)

    assert builder.OUTPUT_COLUMNS == ["Store", "Date", *spec_features]


def test_drops_weeks_without_lag_history_and_keeps_latest_week():
    # Purpose: lag_52 needs 52 earlier weeks, and the latest week has no
    # next-week target but must still be returned for scoring.
    result = builder.build_engineered_features(make_raw())

    assert result.groupby("Store").size().tolist() == [WEEKS - 52, WEEKS - 52]
    assert result["Date"].max() == make_raw()["Date"].max()
    assert not result.isna().any().any()


def test_rows_are_sorted_by_store_then_newest_date_first():
    # Purpose: match the ordering of the offline-store export.
    result = builder.build_engineered_features(make_raw())

    expected = result.sort_values(["Store", "Date"], ascending=[True, False])
    pd.testing.assert_frame_equal(result, expected.reset_index(drop=True))


def test_lag_and_rolling_values_use_previous_weeks_only():
    # Purpose: lag and rolling features must not include the current week.
    raw = make_raw(stores=(1,))
    result = builder.build_engineered_features(raw)
    latest = result.iloc[0]
    sales = raw["Weekly_Sales"]

    assert latest["lag_1"] == sales.iloc[-2]
    assert latest["lag_52"] == sales.iloc[-53]
    assert latest["rollmean_3"] == pytest.approx(sales.iloc[-4:-1].mean())


def test_store_columns_exist_for_stores_missing_from_the_data():
    # Purpose: the model expects S_1 to S_45 even when a store is absent.
    result = builder.build_engineered_features(make_raw(stores=(3,)))

    assert set(builder.STORE_FEATURES).issubset(result.columns)
    assert result["S_3"].all()
    assert not result.drop(columns="S_3").filter(like="S_").any().any()


def test_holiday_flag_maps_to_named_holiday_column():
    # Purpose: a February holiday week becomes Super_Bowl, and only that week.
    raw = make_raw(stores=(1,))
    holiday = pd.Timestamp("2011-02-11")  # has 52 earlier weeks of history
    raw.loc[raw["Date"] == holiday, "Holiday_Flag"] = 1

    result = builder.build_engineered_features(raw)

    assert result.loc[result["Holiday_Super_Bowl"], "Date"].tolist() == [holiday]
    assert not result[["Holiday_Christmas", "Holiday_Thanksgiving", "Holiday_Labor_Day"]].any().any()


def test_main_writes_csv_with_day_first_dates(tmp_path: Path):
    # Purpose: the CLI reads the raw CSV and writes the export file name and
    # date format the rest of the pipeline expects.
    source = tmp_path / "walmart.csv"
    raw = make_raw()
    raw.assign(Date=raw["Date"].dt.strftime("%d-%m-%Y")).to_csv(source, index=False)

    builder.main(["--source", str(source), "--output-dir", str(tmp_path / "out")])

    written = pd.read_csv(tmp_path / "out" / builder.OUTPUT_FILENAME)
    assert list(written.columns) == builder.OUTPUT_COLUMNS
    assert written["Date"].iloc[0] == raw["Date"].max().strftime("%d-%m-%Y")


def test_main_fails_when_no_store_has_enough_history(tmp_path: Path):
    # Purpose: fewer than 53 weeks per store gives an empty export.
    source = tmp_path / "short.csv"
    raw = make_raw(weeks=20)
    raw.assign(Date=raw["Date"].dt.strftime("%d-%m-%Y")).to_csv(source, index=False)

    with pytest.raises(RuntimeError, match="lag history"):
        builder.main(["--source", str(source), "--output-dir", str(tmp_path / "out")])
