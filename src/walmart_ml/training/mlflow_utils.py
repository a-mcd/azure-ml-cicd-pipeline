"""MLflow setup for the training job."""

import os
import mlflow
from walmart_ml.training import config

def configure_mlflow() -> None:
    """Set the MLflow model registry and turn on XGBoost autologging.

    The registry URI comes from the ``MLFLOW_REGISTRY_URI`` environment variable,
    or :data:`config.DEFAULT_REGISTRY_URI` when it is not set. Autologging records
    the model and its signature but not input examples or datasets.
    """
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", config.DEFAULT_REGISTRY_URI))
    mlflow.xgboost.autolog(
        log_models=True,
        log_model_signatures=True,
        log_input_examples=False,
        log_datasets=False,
    )
