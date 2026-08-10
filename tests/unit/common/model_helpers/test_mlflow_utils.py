from types import SimpleNamespace

from  src.common.model_helpers import mlflow_utils


def test_configure_mlflow_uses_default_registry_uri_when_env_var_missing(
    mocker,
    monkeypatch,
):
    # Purpose: Check that configure_mlflow uses the default registry URI
    # when MLFLOW_REGISTRY_URI is not set.
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)

    mock_mlflow = SimpleNamespace(
        set_registry_uri=mocker.Mock(),  # mock_mlflow.set_registry_uri()
        xgboost=SimpleNamespace(
            autolog=mocker.Mock(),  # mock_mlflow.xgboost.autolog()
        ),
    )

    mocker.patch("src.common.model_helpers.mlflow_utils.mlflow", mock_mlflow)
    mocker.patch(
        "src.common.model_helpers.mlflow_utils.config.DEFAULT_REGISTRY_URI",
        "azureml://default-registry-uri",
    )

    mlflow_utils.configure_mlflow()

    mock_mlflow.set_registry_uri.assert_called_once_with(
        "azureml://default-registry-uri"
    )


def test_configure_mlflow_uses_env_registry_uri_when_set(
    mocker,
    monkeypatch,
):
    # Purpose: Check that configure_mlflow uses MLFLOW_REGISTRY_URI
    # when the environment variable is set.
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "azureml://env-registry-uri")

    mock_mlflow = SimpleNamespace(
        set_registry_uri=mocker.Mock(),
        xgboost=SimpleNamespace(
            autolog=mocker.Mock(),
        ),
    )

    mocker.patch("src.common.model_helpers.mlflow_utils.mlflow", mock_mlflow)
    mocker.patch(
        "src.common.model_helpers.mlflow_utils.config.DEFAULT_REGISTRY_URI",
        "azureml://default-registry-uri",
    )

    mlflow_utils.configure_mlflow()

    mock_mlflow.set_registry_uri.assert_called_once_with(
        "azureml://env-registry-uri"
    )


def test_configure_mlflow_enables_xgboost_autolog_with_expected_settings(
    mocker,
    monkeypatch,
):
    # Purpose: Check that configure_mlflow enables XGBoost autologging
    # with the expected MLflow settings.
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)

    mock_mlflow = SimpleNamespace(
        set_registry_uri=mocker.Mock(),
        xgboost=SimpleNamespace(
            autolog=mocker.Mock(),
        ),
    )

    mocker.patch("src.common.model_helpers.mlflow_utils.mlflow", mock_mlflow)

    mlflow_utils.configure_mlflow()

    mock_mlflow.xgboost.autolog.assert_called_once_with(
        log_models=True,
        log_model_signatures=True,
        log_input_examples=False,
        log_datasets=False,
    )


def test_configure_mlflow_returns_none(
    mocker,
    monkeypatch,
):
    # Purpose: Check that configure_mlflow performs setup only
    # and does not return a value.
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)

    mock_mlflow = SimpleNamespace(
        set_registry_uri=mocker.Mock(),
        xgboost=SimpleNamespace(
            autolog=mocker.Mock(),
        ),
    )

    mocker.patch("src.common.model_helpers.mlflow_utils.mlflow", mock_mlflow)

    result = mlflow_utils.configure_mlflow()

    assert result is None