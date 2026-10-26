"""Utilities for accessing the materialized Delta feature store."""

import subprocess

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


HADOOP_VERSION = "3.3.4"


def list_storage_files(
    storage_account: str,
    storage_container: str,
) -> list[str]:
    """Return file paths stored in an Azure Data Lake filesystem.

    Uses the Azure CLI, so the caller must already be signed in.

    Args:
        storage_account: Storage account name.
        storage_container: Filesystem (container) name.

    Returns:
        list[str]: Paths of every file in the filesystem.
    """

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
    """Create a local Delta-enabled Spark session.

    The session can read ``abfss://`` paths in the given storage account using
    its shared key.

    Args:
        application_name: Spark application name.
        storage_account: Storage account to authenticate against.
        storage_key: Shared key for the storage account.

    Returns:
        SparkSession: Session with the Delta and hadoop-azure packages.
    """
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
    """Return unique Delta-table roots from Azure storage file paths.

    Args:
        storage_files: File paths, as returned by :func:`list_storage_files`.

    Returns:
        list[str]: Sorted table root paths, one per ``_delta_log`` directory.
    """
    return sorted(
        {
            path.split("/_delta_log/", maxsplit=1)[0]
            for path in storage_files
            if "/_delta_log/" in path and path.endswith(".json")
        }
    )
