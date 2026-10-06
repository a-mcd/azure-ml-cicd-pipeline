# src/walmart_ml/pipeline/materialized_store.py

"""Utilities for accessing the materialized Delta feature store."""

import subprocess

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


HADOOP_VERSION = "3.3.4"


def list_storage_files(
    storage_account: str,
    storage_container: str,
) -> list[str]:
    """Return file paths stored in an Azure Data Lake filesystem."""
    
    command = [
        "az",
        "storage",
        "fs",
        "file",
        "list",
        "--account-name",
        storage_account,
        "--file-system",
        storage_container,
        "--recursive",
        "true",
        "--exclude-dir",
        "--num-results",
        "5000",
        "--query",
        "[].name",
        "--output",
        "tsv",
        "--only-show-errors",
    ]

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def create_delta_spark_session(
    *,
    application_name: str,
    storage_account: str,
    storage_key: str,
) -> SparkSession:
    """Create a local Delta-enabled Spark session."""
    storage_host = f"{storage_account}.dfs.core.windows.net"

    builder = (
        SparkSession.builder
        .master("local[*]")
        .appName(application_name)
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config(
            f"spark.hadoop.fs.azure.account.auth.type.{storage_host}",
            "SharedKey",
        )
        .config(
            f"spark.hadoop.fs.azure.account.key.{storage_host}",
            storage_key,
        )
    )

    return configure_spark_with_delta_pip(
        builder,
        extra_packages=[
            f"org.apache.hadoop:hadoop-azure:{HADOOP_VERSION}",
        ],
    ).getOrCreate()


def find_delta_roots(storage_files: list[str]) -> list[str]:
    """Return unique Delta-table roots from Azure storage file paths."""
    return sorted(
        {
            path.split("/_delta_log/", maxsplit=1)[0]
            for path in storage_files
            if "/_delta_log/" in path and path.endswith(".json")
        }
    )