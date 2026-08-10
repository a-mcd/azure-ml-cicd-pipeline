"""Walmart sales feature transformations for Azure ML Feature Store."""

from __future__ import annotations

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
)


class WalmartFeatureTransformer(Transformer):
    """
    Create model input features for next-week Walmart sales forecasting.

    The transformation mirrors the existing pandas feature engineering:

    - Calendar features describe the current source row's date.
    - Holiday features describe the current source row's holiday.
    - Lag features use previous sales rows for the same store.
    - Rolling means use previous weeks only and exclude the current row.
    - Rolling means allow partial windows, matching min_periods=1.
    - Store 20 is the omitted one-hot reference category.

    Weekly_Sales_tplus1 is intentionally not returned because it is a
    training target rather than an inference feature.
    """

    ROLL_WINDOWS = (3, 6, 12)
    LAG_PERIODS = (1, 2, 52)
    STORE_NUMBERS = tuple(range(1, 46))

    @keyword_only
    def __init__(self) -> None:
        """Initialise the Azure ML Feature Store transformer."""
        super().__init__()

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        """
        Transform raw Walmart sales rows into feature-store feature rows.

        Args:
            dataframe: Raw Spark DataFrame supplied by the feature-set source.

        Returns:
            Spark DataFrame containing Store, Date and all declared features.
        """
        self._validate_input_columns(dataframe)

        df = self._cast_input_columns(dataframe)

        # One ordered history per store.
        store_window = (
            Window
            .partitionBy("Store")
            .orderBy(F.col("Date"))
        )

        df = self._add_calendar_features(df)
        df = self._add_holiday_features(df)
        df = self._add_time_series_features(df, store_window)
        df = self._drop_rows_without_history(df)
        df = self._add_store_features(df)

        return df.select(*self._output_columns())

    @staticmethod
    def _validate_input_columns(dataframe: DataFrame) -> None:
        """Check that the raw source contains every required column."""
        required_columns = {
            "Store",
            "Date",
            "Weekly_Sales",
            "Holiday_Flag",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
        }

        missing_columns = sorted(
            required_columns.difference(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Raw Walmart data is missing required columns: "
                + ", ".join(missing_columns)
            )

    @classmethod
    def _drop_rows_without_history(
        cls,
        dataframe: DataFrame,
    ) -> DataFrame:
        """
        Remove rows with incomplete historical features.

        Because lag_52 requires 52 previous rows for the same store,
        the first 52 rows for each store will normally be removed.

        Rolling features are also checked so that the returned model
        feature rows contain no missing time-series values.
        """
        required_history_features = [
            *(f"lag_{lag}" for lag in cls.LAG_PERIODS),
            *(f"rollmean_{window}" for window in cls.ROLL_WINDOWS),
        ]

        return dataframe.dropna(
            how="any",
            subset=required_history_features,
        )

    @staticmethod
    def _cast_input_columns(dataframe: DataFrame) -> DataFrame:
        """
        Cast input values to the types required by the transformation.

        The feature-set source specification parses Date using dd-MM-yyyy,
        so Date should already be a timestamp. The extra conversion below
        makes the transformation more defensive during local testing.
        """
        return (
            dataframe
            .withColumn(
                "Store",
                F.col("Store").cast(IntegerType()),
            )
            .withColumn(
                "Date",
                F.to_timestamp(
                    F.col("Date"),
                    "dd-MM-yyyy",
                )
                if dict(dataframe.dtypes).get("Date") == "string"
                else F.col("Date").cast("timestamp"),
            )
            .withColumn(
                "Weekly_Sales",
                F.col("Weekly_Sales").cast(DoubleType()),
            )
            .withColumn(
                "Holiday_Flag",
                F.col("Holiday_Flag").cast(IntegerType()),
            )
            .withColumn(
                "Temperature",
                F.col("Temperature").cast(DoubleType()),
            )
            .withColumn(
                "Fuel_Price",
                F.col("Fuel_Price").cast(DoubleType()),
            )
            .withColumn(
                "CPI",
                F.col("CPI").cast(DoubleType()),
            )
            .withColumn(
                "Unemployment",
                F.col("Unemployment").cast(DoubleType()),
            )
        )

    @staticmethod
    def _add_calendar_features(dataframe: DataFrame) -> DataFrame:
        """
        Add calendar features from the current row's Date.

        This mirrors:

            df["year"] = df["Date"].dt.year
            df["month"] = df["Date"].dt.month
            df["week"] = df["Date"].dt.isocalendar().week
            df["quarter"] = df["Date"].dt.quarter
        """
        return (
            dataframe
            .withColumn(
                "year",
                F.year("Date").cast(IntegerType()),
            )
            .withColumn(
                "month",
                F.month("Date").cast(IntegerType()),
            )
            .withColumn(
                "week",
                F.weekofyear("Date").cast(IntegerType()),
            )
            .withColumn(
                "quarter",
                F.quarter("Date").cast(IntegerType()),
            )
        )

    @staticmethod
    def _add_holiday_features(dataframe: DataFrame) -> DataFrame:
        """
        Add one-hot holiday features.

        This preserves the original pandas rules exactly:

        - Holiday_Flag == 1 and month 11 -> Thanksgiving
        - Holiday_Flag == 1 and month 12 -> Christmas
        - Holiday_Flag == 1 and month 2  -> Super_Bowl
        - Holiday_Flag == 1 and month 9  -> Labor_Day

        The intermediate Holiday string is unnecessary because the feature
        store needs the final one-hot columns.
        """
        is_holiday = F.col("Holiday_Flag") == F.lit(1)
        month = F.col("month")

        return (
            dataframe
            .withColumn(
                "Holiday_Christmas",
                (
                    is_holiday
                    & (month == F.lit(12))
                ).cast(BooleanType()),
            )
            .withColumn(
                "Holiday_Thanksgiving",
                (
                    is_holiday
                    & (month == F.lit(11))
                ).cast(BooleanType()),
            )
            .withColumn(
                "Holiday_Labor_Day",
                (
                    is_holiday
                    & (month == F.lit(9))
                ).cast(BooleanType()),
            )
            .withColumn(
                "Holiday_Super_Bowl",
                (
                    is_holiday
                    & (month == F.lit(2))
                ).cast(BooleanType()),
            )
        )

    @classmethod
    def _add_time_series_features(
        cls,
        dataframe: DataFrame,
        store_window: Window,
    ) -> DataFrame:
        """
        Add per-store lag and rolling-mean features.

        Lag behaviour matches:

            groupby("Store")["Weekly_Sales"].shift(lag)

        Rolling behaviour matches:

            shift(1).rolling(window=w, min_periods=1).mean()

        A Spark frame from -w to -1 excludes the current row and includes up
        to the previous w rows. Spark averages the available non-null rows,
        matching min_periods=1.
        """
        df = dataframe

        for lag in cls.LAG_PERIODS:
            df = df.withColumn(
                f"lag_{lag}",
                F.lag(
                    F.col("Weekly_Sales"),
                    lag,
                )
                .over(store_window)
                .cast(DoubleType()),
            )

        for window_size in cls.ROLL_WINDOWS:
            previous_rows_window = store_window.rowsBetween(
                -window_size,
                -1,
            )

            df = df.withColumn(
                f"rollmean_{window_size}",
                F.avg("Weekly_Sales")
                .over(previous_rows_window)
                .cast(DoubleType()),
            )

        return df

    @classmethod
    def _add_store_features(
        cls,
        dataframe: DataFrame,
    ) -> DataFrame:
        """
        Add one-hot store features.
        """
        df = dataframe

        for store_number in cls.STORE_NUMBERS:

            df = df.withColumn(
                f"S_{store_number}",
                (
                    F.col("Store") == F.lit(store_number)
                ).cast(BooleanType()),
            )

        return df

    @classmethod
    def _store_feature_names(cls) -> list[str]:
        """Return store feature names in model column order."""
        return [
            f"S_{store_number}"
            for store_number in cls.STORE_NUMBERS
        ]

    @classmethod
    def _output_columns(cls) -> list[str]:
        """
        Return timestamp, entity and feature columns.

        Keep this ordering aligned with the model's expected input ordering.
        Store and Date are retained for feature-store indexing and temporal
        retrieval but are not necessarily passed into the trained model.
        """
        return [
            "Store",
            "Date",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
            "lag_1",
            "lag_2",
            "lag_52",
            "rollmean_3",
            "rollmean_6",
            "rollmean_12",
            "month",
            "week",
            "quarter",
            "year",
            *cls._store_feature_names(),
            "Holiday_Christmas",
            "Holiday_Thanksgiving",
            "Holiday_Labor_Day",
            "Holiday_Super_Bowl",
        ]