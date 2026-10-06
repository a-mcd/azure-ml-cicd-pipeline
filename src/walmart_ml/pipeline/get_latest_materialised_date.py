#!/usr/bin/env python3
"""Print the latest date stored in a materialized feature-set Delta table."""

from __future__ import annotations

import os
import sys
import subprocess
from datetime import date, datetime
from pyspark.errors import PySparkException
from pyspark.sql import functions as F
from py4j.protocol import Py4JError
from walmart_ml.common.pipeline_helpers.github_actions import write_environment_variables
from walmart_ml.common.pipeline_helpers.materialized_store import (
    create_delta_spark_session,
    list_storage_files,
    find_delta_roots,
)


HADOOP_VERSION = "3.3.4"
DATE_COLUMN = "Date"
OUTPUT_DATE_FORMAT = "%d-%m-%Y"


def required_environment_variable(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"{name} is not set.")

    return value


def find_materialized_delta_path(
    storage_account: str,
    storage_container: str,
    feature_set_name: str,
    feature_set_version: str,
) -> str:
    """Locate the feature set's materialized Delta table."""

    storage_files = list_storage_files(storage_account, storage_container)
    delta_roots = find_delta_roots(storage_files)

    version_path_component = f"/{feature_set_version}/"
    matching_roots = [
        root
        for root in delta_roots
        if feature_set_name.lower() in root.lower()
        and version_path_component in f"{root}/"
    ]

    if not matching_roots:
        raise RuntimeError(
            "No materialized Delta table was found for "
            f"{feature_set_name} version {feature_set_version}."
        )

    if len(matching_roots) > 1:
        raise RuntimeError(
            "More than one matching Delta table was found:\n"
            + "\n".join(matching_roots)
        )

    storage_host = f"{storage_account}.dfs.core.windows.net"

    return (
        f"abfss://{storage_container}@{storage_host}/"
        f"{matching_roots[0]}"
    )


def format_latest_date(value: date | datetime) -> str:
    """Format the latest materialized date in day-month-year order."""
    return value.strftime(OUTPUT_DATE_FORMAT)


def main() -> None:
    feature_set_name = required_environment_variable("FEATURE_SET_NAME")
    feature_set_version = required_environment_variable("FEATURE_SET_VERSION")
    storage_account = required_environment_variable(
        "OFFLINE_STORAGE_ACCOUNT"
    )
    storage_container = required_environment_variable(
        "OFFLINE_STORAGE_CONTAINER"
    )
    storage_key = required_environment_variable("AZURE_STORAGE_KEY")

    materialized_path = find_materialized_delta_path(
        storage_account=storage_account,
        storage_container=storage_container,
        feature_set_name=feature_set_name,
        feature_set_version=feature_set_version,
    )

    spark = create_delta_spark_session(
        application_name="get-latest-materialized-date",
        storage_account=storage_account,
        storage_key=storage_key,
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        feature_df = (
            spark.read
            .format("delta")
            .load(materialized_path)
        )

        if DATE_COLUMN not in feature_df.columns:
            raise RuntimeError(
                f"Column {DATE_COLUMN!r} was not found in the "
                "materialized Delta table."
            )

        latest_row = (
            feature_df
            .select(
                F.max(
                    F.to_timestamp(F.col(DATE_COLUMN))
                ).alias("latest_date")
            )
            .first()
        )

        latest_date = latest_row["latest_date"] if latest_row else None

        if latest_date is None:
            raise RuntimeError(
                "No valid dates were found in the materialized Delta table."
            )

        print(format_latest_date(latest_date))

        github_env = os.getenv("GITHUB_ENV")
        if not github_env:
            raise RuntimeError(
                "GITHUB_ENV is not set. This script must run inside GitHub Actions."
            )
        write_environment_variables(
            {
                "LATEST_PROCESSED_DATE": format_latest_date(latest_date),
            }
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    try:
        main()
    except (
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        PySparkException,
        Py4JError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
