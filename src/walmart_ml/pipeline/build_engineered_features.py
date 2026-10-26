r"""Build the engineered-feature CSV without the managed feature store.

This is the fallback for when feature-store materialization is unavailable.
It applies the same feature engineering used for model training
(``walmart_ml.training.features`` and ``walmart_ml.training.encoding``) to the
raw Walmart sales CSV, and writes the result in the same format as the
offline-store export produced by ``create_all_weeks_features.py``:

- one row per Store and Date, newest week first within each store
- rows without full lag and rolling-mean history removed
- the latest week kept, because the endpoints score it to predict next week
- Date formatted as DD-MM-YYYY
- columns: Store, Date, then the model features in training order

Usage:
    python -m walmart_ml.pipeline.build_engineered_features \
        --source data/walmart_sales_26-10-2012.csv \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from walmart_ml.logging_config import configure_logging
from walmart_ml.training import data, encoding, features

LOGGER = logging.getLogger(__name__)


OUTPUT_FILENAME = "walmart_materialized_features_all_weeks.csv"
DATE_FORMAT = "%d-%m-%Y"
STORE_IDS = range(1, 46)

BASE_FEATURES = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "lag_1", "lag_2", "lag_52",
    "rollmean_3", "rollmean_6", "rollmean_12",
    "month", "week", "quarter", "year",
]
HISTORY_FEATURES = [
    "lag_1", "lag_2", "lag_52",
    "rollmean_3", "rollmean_6", "rollmean_12",
]
HOLIDAY_FEATURES = [
    "Holiday_Christmas", "Holiday_Thanksgiving",
    "Holiday_Labor_Day", "Holiday_Super_Bowl",
]
STORE_FEATURES = [f"S_{store_id}" for store_id in STORE_IDS]
OUTPUT_COLUMNS = ["Store", "Date", *BASE_FEATURES, *STORE_FEATURES, *HOLIDAY_FEATURES]


def build_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature rows for every store and week with history.

    Args:
        raw: Raw Walmart sales data with Date already parsed as datetime.

    Returns:
        pd.DataFrame: Rows with OUTPUT_COLUMNS, Date still as datetime.

    Raises:
        ValueError: If feature engineering did not produce every expected column.
    """
    # Keep rows without a next-week target: the latest week is the one the
    # endpoints score.
    engineered = features.build_features(raw, drop_training_rows=False)
    engineered = engineered.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)

    store_ids = engineered["Store"].astype(int)
    encoded = encoding.prepare_encodings(engineered)

    # get_dummies only creates columns for stores present in the data.
    encoded = encoded.reindex(
        columns=[*encoded.columns, *(c for c in STORE_FEATURES if c not in encoded.columns)],
        fill_value=False,
    )

    missing = sorted(set(BASE_FEATURES + HOLIDAY_FEATURES).difference(encoded.columns))
    if missing:
        raise ValueError(f"Feature engineering did not produce columns: {missing}")

    encoded["Store"] = store_ids
    encoded[STORE_FEATURES + HOLIDAY_FEATURES] = encoded[STORE_FEATURES + HOLIDAY_FEATURES].astype(bool)

    return (
        encoded[OUTPUT_COLUMNS]
        .sort_values(["Store", "Date"], ascending=[True, False])
        .reset_index(drop=True)
    )


def write_engineered_features(engineered: pd.DataFrame, output_dir: Path) -> Path:
    """Write the engineered features CSV.

    Args:
        engineered: Output of :func:`build_engineered_features`.
        output_dir: Folder to write the CSV into. Created if it does not exist.

    Returns:
        Path: Path of the CSV that was written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    export = engineered.copy()
    export["Date"] = export["Date"].dt.strftime(DATE_FORMAT)
    export.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed ``source`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Build the engineered-feature CSV without the managed feature store."
    )
    parser.add_argument("--source", required=True, help="Raw Walmart sales CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=f"Folder to write {OUTPUT_FILENAME} into.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build the engineered features from a raw CSV and write them to disk.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Raises:
        RuntimeError: If no rows have enough history to build lag features.
    """
    configure_logging()
    args = parse_args(argv)

    raw = data.load_dataset(args.source)
    engineered = build_engineered_features(raw)
    if engineered.empty:
        raise RuntimeError("No rows have full lag history; the source needs at least 53 weeks per store.")

    output_path = write_engineered_features(engineered, args.output_dir)

    LOGGER.info(
        "Wrote %s rows for %s stores to %s",
        f"{len(engineered):,}",
        engineered["Store"].nunique(),
        output_path,
    )
    LOGGER.info(
        "Date range: %s to %s",
        engineered["Date"].min().strftime(DATE_FORMAT),
        engineered["Date"].max().strftime(DATE_FORMAT),
    )


if __name__ == "__main__":
    main()
