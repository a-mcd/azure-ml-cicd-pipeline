"""Export all weeks per store from the offline Delta store."""

import os
from pathlib import Path
from pyspark.sql import functions as F
from src.common.pipeline_helpers.materialized_store import (
    create_delta_spark_session,
    find_delta_roots,
    list_storage_files,
)

FEATURE_SET_NAME = os.getenv("FEATURE_SET_NAME")
FEATURE_SET_VERSION = os.getenv("FEATURE_SET_VERSION")

HADOOP_VERSION = "3.3.4"

OFFLINE_STORAGE_ACCOUNT = os.getenv("OFFLINE_STORAGE_ACCOUNT")
OFFLINE_STORAGE_CONTAINER = os.getenv("OFFLINE_STORAGE_CONTAINER")

OUTPUT_PATH = Path(
    "data/walmart_materialized_features_all_weeks.csv"
)


def require_environment_variable(
    variable_name: str,
    variable_value: str | None,
) -> str:
    """Return a required environment variable or raise an error."""
    if not variable_value:
        raise RuntimeError(f"{variable_name} is not set.")

    return variable_value


FEATURE_SET_NAME = require_environment_variable(
    "FEATURE_SET_NAME",
    FEATURE_SET_NAME,
)

FEATURE_SET_VERSION = require_environment_variable(
    "FEATURE_SET_VERSION",
    FEATURE_SET_VERSION,
)

OFFLINE_STORAGE_ACCOUNT = require_environment_variable(
    "OFFLINE_STORAGE_ACCOUNT",
    OFFLINE_STORAGE_ACCOUNT,
)

OFFLINE_STORAGE_CONTAINER = require_environment_variable(
    "OFFLINE_STORAGE_CONTAINER",
    OFFLINE_STORAGE_CONTAINER,
)

OFFLINE_STORAGE_HOST = (
    f"{OFFLINE_STORAGE_ACCOUNT}.dfs.core.windows.net"
)


def find_materialized_delta_path() -> str:
    """Locate the materialised Delta table using its _delta_log directory."""

    file_paths = list_storage_files(OFFLINE_STORAGE_ACCOUNT, OFFLINE_STORAGE_CONTAINER)
    delta_roots = find_delta_roots(file_paths)

    print("Delta tables found in the offline store:")

    for delta_root in delta_roots:
        print(f"  {delta_root}")

    matching_roots = [
        root
        for root in delta_roots
        if FEATURE_SET_NAME.lower() in root.lower()
        and f"/{FEATURE_SET_VERSION}" in root
    ]

    if not matching_roots:
        raise RuntimeError(
            "No materialised Delta table was found for "
            f"{FEATURE_SET_NAME} version {FEATURE_SET_VERSION}. "
            "Review the Delta table paths printed above."
        )

    if len(matching_roots) > 1:
        raise RuntimeError(
            "More than one matching Delta table was found:\n"
            + "\n".join(matching_roots)
        )

    relative_path = matching_roots[0]

    return (
        f"abfss://{OFFLINE_STORAGE_CONTAINER}"
        f"@{OFFLINE_STORAGE_HOST}/"
        f"{relative_path}"
    )


def main() -> None:
    """Export all materialised feature rows to CSV."""
    storage_key = require_environment_variable(
        "AZURE_STORAGE_KEY",
        os.getenv("AZURE_STORAGE_KEY"),
    )

    spark = create_delta_spark_session(
        application_name="export-walmart-all-weeks",
        storage_account=OFFLINE_STORAGE_ACCOUNT,
        storage_key=storage_key,
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        materialized_path = find_materialized_delta_path()

        print(
            "\nReading materialised features from:\n"
            f"{materialized_path}"
        )

        feature_df = (
            spark.read
            .format("delta")
            .load(materialized_path)
        )

        print("\nMaterialised feature schema:")
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
                "No materialised feature rows with valid dates were found."
            )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\nWriting all materialised weeks for every store to CSV..."
        )

        export_df = all_weeks_df.withColumn(
            "Date",
            F.date_format(
                F.col("Date"),
                "dd-MM-yyyy",
            ),
        )

        (
            export_df
            .toPandas()
            .to_csv(
                OUTPUT_PATH,
                index=False,
            )
        )

        print(f"CSV written to: {OUTPUT_PATH.resolve()}")

        print("\nMaterialised date range:")
        (
            all_weeks_df
            .agg(
                F.min("Date").alias("Earliest_Date"),
                F.max("Date").alias("Latest_Date"),
            )
            .show(truncate=False)
        )

        print("\nNumber of exported rows per store:")
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

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
