"""Export all weeks per store from the offline Delta store.

Settings are read from environment variables when :func:`main` runs, not when
the module is imported:

    FEATURE_SET_NAME: Feature set whose materialized table to export.
    FEATURE_SET_VERSION: Feature set version.
    OFFLINE_STORAGE_ACCOUNT: Storage account of the offline store.
    OFFLINE_STORAGE_CONTAINER: Container (filesystem) of the offline store.
    AZURE_STORAGE_KEY: Shared key for the storage account.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from walmart_ml.logging_config import configure_logging
from walmart_ml.pipeline.materialized_store import (
    create_delta_spark_session,
    find_delta_roots,
    list_storage_files,
)

LOGGER = logging.getLogger(__name__)

OUTPUT_PATH = Path(
    "data/walmart_materialized_features_all_weeks.csv"
)
SPARK_APPLICATION_NAME = "export-walmart-all-weeks"


@dataclass(frozen=True)
class ExportSettings:
    """Where the materialized feature table lives and how to reach it.

    Attributes:
        feature_set_name: Feature set whose table to export.
        feature_set_version: Feature set version.
        storage_account: Storage account of the offline store.
        storage_container: Container (filesystem) of the offline store.
        storage_key: Shared key for the storage account.
    """

    feature_set_name: str
    feature_set_version: str
    storage_account: str
    storage_container: str
    storage_key: str

    @property
    def storage_host(self) -> str:
        """str: Data Lake host name of the storage account."""
        return f"{self.storage_account}.dfs.core.windows.net"


def require_environment_variable(name: str) -> str:
    """Return a required environment variable or raise an error.

    Args:
        name: Name of the environment variable.

    Returns:
        str: The variable's value.

    Raises:
        RuntimeError: If the variable is missing or empty.
    """
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"{name} is not set.")

    return value


def load_settings() -> ExportSettings:
    """Read the export settings from environment variables.

    Returns:
        ExportSettings: Settings for this run.

    Raises:
        RuntimeError: If any required variable is missing or empty.
    """
    return ExportSettings(
        feature_set_name=require_environment_variable("FEATURE_SET_NAME"),
        feature_set_version=require_environment_variable("FEATURE_SET_VERSION"),
        storage_account=require_environment_variable("OFFLINE_STORAGE_ACCOUNT"),
        storage_container=require_environment_variable("OFFLINE_STORAGE_CONTAINER"),
        storage_key=require_environment_variable("AZURE_STORAGE_KEY"),
    )


def find_materialized_delta_path(settings: ExportSettings) -> str:
    """Locate the materialized Delta table using its _delta_log directory.

    Args:
        settings: Feature set and storage location to search.

    Returns:
        str: ``abfss://`` URI of the feature set's Delta table.

    Raises:
        RuntimeError: If no matching table, or more than one, is found.
    """
    file_paths = list_storage_files(settings.storage_account, settings.storage_container)
    delta_roots = find_delta_roots(file_paths)

    LOGGER.info(
        "Delta tables found in the offline store:\n%s",
        "\n".join(f"  {delta_root}" for delta_root in delta_roots),
    )

    matching_roots = [
        root
        for root in delta_roots
        if settings.feature_set_name.lower() in root.lower()
        and f"/{settings.feature_set_version}" in root
    ]

    if not matching_roots:
        raise RuntimeError(
            "No materialized Delta table was found for "
            f"{settings.feature_set_name} version {settings.feature_set_version}. "
            "Review the Delta table paths logged above."
        )

    if len(matching_roots) > 1:
        raise RuntimeError(
            "More than one matching Delta table was found:\n"
            + "\n".join(matching_roots)
        )

    return (
        f"abfss://{settings.storage_container}"
        f"@{settings.storage_host}/"
        f"{matching_roots[0]}"
    )


def read_all_weeks(spark: SparkSession, materialized_path: str) -> DataFrame:
    """Read the Delta table and keep the rows with a valid date.

    Args:
        spark: Delta-enabled Spark session.
        materialized_path: Path or URI of the Delta table.

    Returns:
        DataFrame: Feature rows with Date as a date, ordered by Store and
        newest week first.

    Raises:
        RuntimeError: If no rows have a valid date.
    """
    LOGGER.info("Reading materialized features from: %s", materialized_path)

    feature_df = (
        spark.read
        .format("delta")
        .load(materialized_path)
    )

    LOGGER.info("Materialized feature schema:")
    feature_df.printSchema()

    # Convert Date to a date value so rows are ordered chronologically,
    # rather than lexicographically as strings.
    all_weeks_df = (
        feature_df
        .withColumn(
            "_parsed_date",
            F.to_date(F.col("Date")),
        )
        .filter(F.col("_parsed_date").isNotNull())
        .drop("Date")
        .withColumnRenamed("_parsed_date", "Date")
        .orderBy("Store", F.col("Date").desc())
    )

    if all_weeks_df.limit(1).count() == 0:
        raise RuntimeError(
            "No materialized feature rows with valid dates were found."
        )

    return all_weeks_df


def write_all_weeks_csv(all_weeks_df: DataFrame, output_path: Path) -> None:
    """Write the feature rows to CSV with dates as DD-MM-YYYY.

    Args:
        all_weeks_df: Output of :func:`read_all_weeks`.
        output_path: CSV file to write. Its parent folder is created.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info("Writing all materialized weeks for every store to CSV...")

    export_df = all_weeks_df.withColumn(
        "Date",
        F.date_format(
            F.col("Date"),
            "dd-MM-yyyy",
        ),
    )

    # pyspark annotates toPandas() with a pandas protocol mypy cannot resolve,
    # so treat the result as an untyped pandas DataFrame.
    pandas_export: Any = export_df.toPandas()
    pandas_export.to_csv(output_path, index=False)

    LOGGER.info("CSV written to: %s", output_path.resolve())


def show_export_summary(all_weeks_df: DataFrame) -> None:
    """Show the exported date range and the row count for each store.

    Args:
        all_weeks_df: Output of :func:`read_all_weeks`.
    """
    LOGGER.info("Materialized date range:")
    (
        all_weeks_df
        .agg(
            F.min("Date").alias("Earliest_Date"),
            F.max("Date").alias("Latest_Date"),
        )
        .show(truncate=False)
    )

    LOGGER.info("Number of exported rows per store:")
    (
        all_weeks_df
        .groupBy("Store")
        .agg(
            F.count("*").alias("Row_Count"),
            F.countDistinct("Date").alias("Week_Count"),
            F.min("Date").alias("Earliest_Date"),
            F.max("Date").alias("Latest_Date"),
        )
        .orderBy("Store")
        .show(
            n=100,
            truncate=False,
        )
    )


def main() -> None:
    """Export all materialized feature rows to CSV.

    Raises:
        RuntimeError: If a required environment variable is missing, or the
            table has no rows with valid dates.
    """
    configure_logging()
    settings = load_settings()

    spark = create_delta_spark_session(
        application_name=SPARK_APPLICATION_NAME,
        storage_account=settings.storage_account,
        storage_key=settings.storage_key,
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        materialized_path = find_materialized_delta_path(settings)
        all_weeks_df = read_all_weeks(spark, materialized_path)
        write_all_weeks_csv(all_weeks_df, OUTPUT_PATH)
        show_export_summary(all_weeks_df)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
