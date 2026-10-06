import os
import mlflow
from walmart_ml.common.model_helpers import config

def configure_mlflow() -> None:
    mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", config.DEFAULT_REGISTRY_URI))
    mlflow.xgboost.autolog(
        log_models=True,
        log_model_signatures=True,
        log_input_examples=False,
        log_datasets=False,
    )
