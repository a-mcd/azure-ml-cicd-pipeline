"""Check the pandas fallback matches the feature store's Spark transformer.

build_engineered_features.py replaces feature-store materialization when it
is unavailable. Both must produce the same feature values, or the endpoints
would score different inputs depending on which path built the data.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from walmart_ml.pipeline import build_engineered_features as builder


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMER_PATH = (
    PROJECT_ROOT
    / "azureml"
    / "feature_store"
    / "feature_sets"
    / "walmart_sales_features"
    / "spec"
    / "transformation_code"
    / "walmart_transformer.py"
)
SOURCE_CSV = PROJECT_ROOT / "data" / "walmart_sales_26-10-2012.csv"


@pytest.fixture(scope="module")
def spark():
    # Start Spark with the Delta jars even though this test doesn't use Delta.
    # Spark loads jars only once per process, so if this test started a plain
    # session first, the Delta tests that run after it would fail with
    # "Failed to find the data source: delta".
    builder = (
        SparkSession.builder
        .master("local[1]")
        .appName("engineered-features-parity")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def _load_transformer():
    spec = importlib.util.spec_from_file_location("walmart_transformer", TRANSFORMER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WalmartFeatureTransformer()


def test_pandas_fallback_matches_feature_store_transformer(spark):
    raw_csv = pd.read_csv(SOURCE_CSV)

    spark_result = (
        _load_transformer()
        .transform(spark.createDataFrame(raw_csv))
        .toPandas()
    )
    spark_result["Date"] = pd.to_datetime(spark_result["Date"]).dt.normalize()
    spark_result = (
        spark_result[builder.OUTPUT_COLUMNS]
        .sort_values(["Store", "Date"], ascending=[True, False])
        .reset_index(drop=True)
    )

    raw = raw_csv.assign(Date=pd.to_datetime(raw_csv["Date"], format="%d-%m-%Y"))
    pandas_result = builder.build_engineered_features(raw)

    assert len(pandas_result) == len(spark_result)
    pd.testing.assert_series_equal(pandas_result["Store"], spark_result["Store"], check_dtype=False)
    pd.testing.assert_series_equal(pandas_result["Date"], spark_result["Date"], check_dtype=False)

    boolean_columns = builder.STORE_FEATURES + builder.HOLIDAY_FEATURES
    for column in boolean_columns:
        assert (pandas_result[column].astype(bool) == spark_result[column].astype(bool)).all(), column

    for column in builder.BASE_FEATURES:
        assert np.allclose(
            pandas_result[column].astype(float),
            spark_result[column].astype(float),
        ), column
