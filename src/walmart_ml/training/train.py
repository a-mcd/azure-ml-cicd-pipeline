"""Azure ML command job that trains the XGBoost sales model.

Builds features, splits the data by date, trains the model inside an MLflow
run and writes evaluation outputs. With ``--challenger_model`` it holds out a
single week and saves a forecast for the champion-challenger comparison.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from walmart_ml.training import (
    config,
    data,
    encoding,
    features,
    metrics,
    mlflow_utils,
    model,
    outputs,
    validation
)
from walmart_ml.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Encoded model inputs and targets for one train/test split.

    Attributes:
        feature_cols: Model input columns, in training order.
        X_train: Training features.
        y_train: Training targets (next week's sales).
        X_test: Test features.
        y_test: Test targets.
        test: Test rows before encoding, used for the challenger forecast.
    """

    feature_cols: list[str]
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse the training job's command-line arguments.

    Returns:
        argparse.Namespace: Training data path, XGBoost hyperparameters and
        the ``challenger_model`` flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True, help='Dataset for training')
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument(
        "--challenger_model",
        action="store_true",
        help="Train a challenger model for comparison with the production model.",
    )
    return parser.parse_args()


def get_test_horizon_weeks(challenger_model: bool) -> int:
    """Return how many of the latest weeks to hold out for testing.

    Args:
        challenger_model: Whether this run trains a challenger model.

    Returns:
        int: ``CHALLENGER_HORIZON_WEEKS`` for a challenger, otherwise
        ``TEST_HORIZON_WEEKS``.
    """
    if challenger_model:
        test_horizon_weeks = config.CHALLENGER_HORIZON_WEEKS
        LOGGER.info("Challenger model: using %s week(s) for test set.", test_horizon_weeks)
    else:
        test_horizon_weeks = config.TEST_HORIZON_WEEKS
        LOGGER.info("Candidate model: using %s weeks for test set.", test_horizon_weeks)
    return test_horizon_weeks


def prepare_data(df_feat: pd.DataFrame, test_horizon_weeks: int, outputs_dir: Path) -> PreparedData:
    """Split, encode and validate the data, and save the train and test sets.

    Args:
        df_feat: Dataset after feature engineering.
        test_horizon_weeks: Number of latest weeks to use for testing.
        outputs_dir: Directory for the train and test CSVs.

    Returns:
        PreparedData: Model inputs and targets for training and testing.
    """
    train, test = data.chronological_split(df_feat, test_horizon_weeks)

    # Keep metadata for output BEFORE encoding
    train_meta = train[config.METADATA_COLUMNS].copy()
    test_meta = test[config.METADATA_COLUMNS].copy()

    # One-hot encode
    train_enc = encoding.prepare_encodings(train)
    test_enc = encoding.prepare_encodings(test)

    # Build the final list of columns that will be used as model input features.
    feature_cols = encoding.get_feature_columns(train_enc)
    required_cols = feature_cols + ["Weekly_Sales_tplus1"]
    LOGGER.info("Features: %s", f"{len(required_cols):,}")
    LOGGER.info("%s", required_cols)

    validation.validate_missing_values(train_enc, "training", required_cols)
    validation.validate_missing_values(test_enc, "test", required_cols)

    # Separate features and target
    prepared = PreparedData(
        feature_cols=feature_cols,
        X_train=train_enc[feature_cols].copy(),
        y_train=train_enc["Weekly_Sales_tplus1"].copy(),
        X_test=test_enc[feature_cols].copy(),
        y_test=test_enc["Weekly_Sales_tplus1"].copy(),
        test=test,
    )
    LOGGER.info("Training rows: %s", f"{len(prepared.X_train):,}")
    LOGGER.info("Test rows: %s", f"{len(prepared.X_test):,}")

    outputs.output_train_test_sets(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        train_meta=train_meta,
        test_meta=test_meta,
        outputs_dir=outputs_dir,
    )

    return prepared


def log_run_params(args: argparse.Namespace, test_horizon_weeks: int, prepared: PreparedData) -> None:
    """Record the dataset and split details in the active MLflow run.

    Args:
        args: Parsed training arguments.
        test_horizon_weeks: Number of weeks held out for testing.
        prepared: Output of :func:`prepare_data`.
    """
    mlflow.log_param("training_data", os.path.abspath(args.training_data))
    mlflow.log_param("test_horizon_weeks", test_horizon_weeks)
    mlflow.log_param("challenger_model", args.challenger_model)
    mlflow.log_param("roll_windows", ",".join(map(str, config.ROLL_WINDOWS)))
    mlflow.log_param("n_features", len(prepared.feature_cols))
    mlflow.log_param("n_train_rows", int(len(prepared.X_train)))
    mlflow.log_param("n_test_rows", int(len(prepared.X_test)))


def train_and_evaluate(args: argparse.Namespace, prepared: PreparedData, outputs_dir: Path) -> None:
    """Train the model, evaluate it on the test set and save its outputs.

    Must run inside an active MLflow run.

    Args:
        args: Parsed training arguments, including the hyperparameters.
        prepared: Output of :func:`prepare_data`.
        outputs_dir: Directory for the evaluation outputs.
    """
    xgb_model = model.create_xgb_model(args)

    # Log model hyperparameters
    mlflow.log_params(xgb_model.get_params())

    xgb_model.fit(prepared.X_train, prepared.y_train)

    # In future version
    # - train directly from the feature store
    # - Create the feature_retrieval_spec.yaml dynamically and add it as as artifact
    # spec_path = "feature_retrieval_spec/feature_retrieval_spec.yaml"
    # mlflow.log_artifact(
    #     local_path=spec_path,
    #     artifact_path="model"
    # )

    preds = xgb_model.predict(prepared.X_test)

    metrics.calculate_regression_metrics(prepared.y_test, preds)

    # mlflow.log_artifacts not used as there is a bug.
    outputs.output_feature_importance(xgb_model, prepared.X_train, outputs_dir)

    if args.challenger_model:
        outputs.output_challenger_forecast_by_store(
            test=prepared.test,
            predictions=preds,
            outputs_dir=outputs_dir,
        )


def main() -> None:
    """Parse the arguments, then train, evaluate and log the model."""
    configure_logging()
    args = parse_args()

    outputs_dir = data.get_outputs_dir()

    df = data.load_dataset(args.training_data)
    df_feat = features.build_features(df)

    test_horizon_weeks = get_test_horizon_weeks(args.challenger_model)
    prepared = prepare_data(df_feat, test_horizon_weeks, outputs_dir)

    mlflow_utils.configure_mlflow()
    with mlflow.start_run():
        log_run_params(args, test_horizon_weeks, prepared)
        train_and_evaluate(args, prepared, outputs_dir)


if __name__ == "__main__":
    main()
