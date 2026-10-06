"""Unit tests for MLflow model metric validation."""

import json
import sys
from pathlib import Path

import pytest

from walmart_ml.pipeline_scripts import validate_model_metrics as validation


@pytest.fixture
def valid_metrics() -> dict:
    """Return model metrics that pass the default thresholds."""
    return {
        "R2": 0.95,
        "MAPE": 8.0,
        "MAE": 100.0,
        "RMSE": 150.0,
        "Bias": 10.0,
        "WAPE": 7.5,
        "Median_AE": 75.0,
    }


@pytest.fixture
def thresholds() -> dict:
    """Return model validation thresholds."""
    return {
        "R2_MIN": 0.90,
        "MAPE_MAX": 10.0,
        "MAE_MAX": 100000.0,
        "RMSE_RATIO_MAX": 2.0,
    }


@pytest.fixture
def valid_params() -> dict:
    """Return valid model hyperparameters."""
    return {
        "learning_rate": "0.05",
        "max_depth": "8",
        "n_estimators": "600",
        "subsample": "0.8",
        "colsample_bytree": "0.8",
        "reg_alpha": "0.0",
        "reg_lambda": "1.0",
    }


@pytest.fixture
def mlflow_response(valid_metrics, valid_params) -> dict:
    """Return a representative MLflow run response."""
    return {
        "run": {
            "data": {
                "metrics": [
                    {"key": key, "value": value}
                    for key, value in valid_metrics.items()
                ],
                "params": [
                    {"key": key, "value": value}
                    for key, value in valid_params.items()
                ],
            }
        }
    }


def test_mlflow_items_to_dict_converts_items():
    items = [
        {"key": "R2", "value": 0.95},
        {"key": "MAPE", "value": 8.0},
    ]

    result = validation.mlflow_items_to_dict(items)

    assert result == {
        "R2": 0.95,
        "MAPE": 8.0,
    }


def test_mlflow_items_to_dict_ignores_items_without_key():
    items = [
        {"key": "R2", "value": 0.95},
        {"value": 8.0},
        {},
    ]

    result = validation.mlflow_items_to_dict(items)

    assert result == {"R2": 0.95}


def test_mlflow_items_to_dict_uses_none_for_missing_value():
    items = [{"key": "R2"}]

    result = validation.mlflow_items_to_dict(items)

    assert result == {"R2": None}


def test_mlflow_items_to_dict_accepts_empty_list():
    assert validation.mlflow_items_to_dict([]) == {}


def test_run_command_returns_stripped_stdout(mocker):
    completed_process = mocker.Mock(
        stdout="  command output\n"
    )
    mock_run = mocker.patch.object(
        validation.subprocess,
        "run",
        return_value=completed_process,
    )

    result = validation.run_command(["example", "--option"])

    assert result == "command output"
    mock_run.assert_called_once_with(
        ["example", "--option"],
        check=True,
        text=True,
        capture_output=True,
    )


def test_run_command_propagates_command_failure(mocker):
    mocker.patch.object(
        validation.subprocess,
        "run",
        side_effect=validation.subprocess.CalledProcessError(
            returncode=1,
            cmd=["example"],
        ),
    )

    with pytest.raises(validation.subprocess.CalledProcessError):
        validation.run_command(["example"])


def test_get_tracking_uri_builds_command_and_normalises_uri(
    mocker,
):
    mock_run_command = mocker.patch.object(
        validation,
        "run_command",
        return_value=(
            "azureml://example.api.azureml.ms/mlflow/v1.0?"
        ),
    )

    result = validation.get_tracking_uri(
        job_name="training-job",
        resource_group="test-rg",
        workspace_name="test-workspace",
    )

    assert result == (
        "https://example.api.azureml.ms/mlflow/v1.0"
    )

    mock_run_command.assert_called_once_with(
        [
            "az",
            "ml",
            "job",
            "show",
            "--name",
            "training-job",
            "--resource-group",
            "test-rg",
            "--workspace-name",
            "test-workspace",
            "--query",
            "services.Tracking.endpoint",
            "-o",
            "tsv",
        ]
    )


def test_get_access_token_builds_expected_command(mocker):
    mock_run_command = mocker.patch.object(
        validation,
        "run_command",
        return_value="access-token",
    )

    result = validation.get_access_token()

    assert result == "access-token"
    mock_run_command.assert_called_once_with(
        [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://ml.azure.com",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ]
    )


def test_get_mlflow_run_json_queries_expected_url(mocker):
    response = {
        "run": {
            "data": {
                "metrics": [],
                "params": [],
            }
        }
    }
    mock_run_command = mocker.patch.object(
        validation,
        "run_command",
        return_value=json.dumps(response),
    )

    result = validation.get_mlflow_run_json(
        tracking_uri="https://tracking.example",
        job_name="training-job",
        token="access-token",
    )

    assert result == response
    mock_run_command.assert_called_once_with(
        [
            "curl",
            "-sS",
            "-H",
            "Authorization: Bearer access-token",
            (
                "https://tracking.example/api/2.0/mlflow/"
                "runs/get?run_id=training-job"
            ),
        ]
    )


def test_get_mlflow_run_json_rejects_invalid_json(mocker):
    mocker.patch.object(
        validation,
        "run_command",
        return_value="not-json",
    )

    with pytest.raises(json.JSONDecodeError):
        validation.get_mlflow_run_json(
            tracking_uri="https://tracking.example",
            job_name="training-job",
            token="access-token",
        )


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [
        (9.0, 10.0, True),
        (10.0, 10.0, True),
        (11.0, 10.0, False),
    ],
)
def test_check_maximum_returns_expected_result(
    value,
    threshold,
    expected,
):
    result = validation.check_maximum(
        "MAPE",
        value,
        threshold,
    )

    assert result is expected


def test_check_maximum_prints_passed_notice(capsys):
    result = validation.check_maximum(
        "MAPE",
        8.0,
        10.0,
        debug=True,
    )

    output = capsys.readouterr().out

    assert result is True
    assert "::notice title=MAPE passed::" in output
    assert "MAPE=8.0, maximum=10.0" in output


def test_check_maximum_prints_failed_warning(capsys):
    result = validation.check_maximum(
        "MAPE",
        12.0,
        10.0,
        debug=True,
    )

    output = capsys.readouterr().out

    assert result is False
    assert "::warning title=MAPE failed::" in output
    assert "MAPE=12.0, maximum=10.0" in output


def test_check_maximum_prints_nothing_when_debug_is_false(
    capsys,
):
    validation.check_maximum(
        "MAPE",
        8.0,
        10.0,
        debug=False,
    )

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("value", "threshold", "expected"),
    [
        (0.95, 0.90, True),
        (0.90, 0.90, True),
        (0.85, 0.90, False),
    ],
)
def test_check_minimum_returns_expected_result(
    value,
    threshold,
    expected,
):
    result = validation.check_minimum(
        "R2",
        value,
        threshold,
    )

    assert result is expected


def test_check_minimum_prints_passed_notice(capsys):
    result = validation.check_minimum(
        "R2",
        0.95,
        0.90,
        debug=True,
    )

    output = capsys.readouterr().out

    assert result is True
    assert "::notice title=R2 passed::" in output
    assert "R2=0.95, minimum=0.9" in output


def test_check_minimum_prints_failed_warning(capsys):
    result = validation.check_minimum(
        "R2",
        0.85,
        0.90,
        debug=True,
    )

    output = capsys.readouterr().out

    assert result is False
    assert "::warning title=R2 failed::" in output
    assert "R2=0.85, minimum=0.9" in output


def test_check_minimum_prints_nothing_when_debug_is_false(
    capsys,
):
    validation.check_minimum(
        "R2",
        0.95,
        0.90,
        debug=False,
    )

    assert capsys.readouterr().out == ""


def test_validate_model_metrics_returns_passing_result(
    valid_metrics,
    thresholds,
):
    passed, result = validation.validate_model_metrics(
        valid_metrics,
        thresholds,
    )

    assert passed is True
    assert result == {
        "R2": 0.95,
        "MAPE": 8.0,
        "MAE": 100.0,
        "RMSE": 150.0,
        "RMSE_RATIO": 1.5,
        "BIAS": 10.0,
        "WAPE": 7.5,
        "MEDIAN_AE": 75.0,
    }


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("R2", 0.85),
        ("MAPE", 11.0),
        ("MAE", 100001.0),
        ("RMSE", 201.0),
    ],
)
def test_validate_model_metrics_returns_failure_when_gate_fails(
    valid_metrics,
    thresholds,
    metric,
    value,
):
    valid_metrics[metric] = value

    passed, _ = validation.validate_model_metrics(
        valid_metrics,
        thresholds,
    )

    assert passed is False


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("R2", 0.90),
        ("MAPE", 10.0),
        ("MAE", 100000.0),
    ],
)
def test_validate_model_metrics_accepts_threshold_boundaries(
    valid_metrics,
    thresholds,
    metric,
    value,
):
    valid_metrics[metric] = value

    if metric == "MAE":
        valid_metrics["RMSE"] = 200000.0

    passed, _ = validation.validate_model_metrics(
        valid_metrics,
        thresholds,
    )

    assert passed is True


@pytest.mark.parametrize(
    ("missing_metric", "missing_value"),
    [
        ("R2", None),
        ("MAPE", ""),
        ("MAE", None),
        ("RMSE", ""),
    ],
)
def test_validate_model_metrics_rejects_missing_values(
    valid_metrics,
    thresholds,
    missing_metric,
    missing_value,
):
    valid_metrics[missing_metric] = missing_value

    with pytest.raises(
        ValueError,
        match="Missing required metrics",
    ):
        validation.validate_model_metrics(
            valid_metrics,
            thresholds,
        )


def test_validate_model_metrics_rejects_absent_metric(
    valid_metrics,
    thresholds,
):
    del valid_metrics["R2"]

    with pytest.raises(
        ValueError,
        match="Missing required metrics: \\['R2'\\]",
    ):
        validation.validate_model_metrics(
            valid_metrics,
            thresholds,
        )


def test_validate_model_metrics_rejects_zero_mae(
    valid_metrics,
    thresholds,
):
    valid_metrics["MAE"] = 0

    with pytest.raises(
        ValueError,
        match="MAE is zero",
    ):
        validation.validate_model_metrics(
            valid_metrics,
            thresholds,
        )


def test_validate_model_metrics_converts_string_values(
    valid_metrics,
    thresholds,
):
    string_metrics = {
        key: str(value)
        for key, value in valid_metrics.items()
    }

    passed, result = validation.validate_model_metrics(
        string_metrics,
        thresholds,
    )

    assert passed is True
    assert result["R2"] == 0.95
    assert result["MAPE"] == 8.0
    assert result["MAE"] == 100.0
    assert result["RMSE"] == 150.0
    assert result["RMSE_RATIO"] == 1.5


def test_validate_model_metrics_passes_debug_to_checks(
    mocker,
    valid_metrics,
    thresholds,
):
    mock_minimum = mocker.patch.object(
        validation,
        "check_minimum",
        return_value=True,
    )
    mock_maximum = mocker.patch.object(
        validation,
        "check_maximum",
        return_value=True,
    )

    passed, _ = validation.validate_model_metrics(
        valid_metrics,
        thresholds,
        debug=True,
    )

    assert passed is True

    mock_minimum.assert_called_once_with(
        "R2",
        0.95,
        0.90,
        debug=True,
    )

    assert mock_maximum.call_args_list == [
        mocker.call(
            "MAPE",
            8.0,
            10.0,
            debug=True,
        ),
        mocker.call(
            "MAE",
            100.0,
            100000.0,
            debug=True,
        ),
        mocker.call(
            "RMSE_RATIO",
            1.5,
            2.0,
            debug=True,
        ),
    ]


def test_build_eval_result_builds_expected_result(
    valid_params,
):
    metrics_result = {
        "R2": 0.95,
        "MAPE": 8.0,
        "MAE": 100.0,
        "RMSE": 150.0,
        "RMSE_RATIO": 1.5,
        "BIAS": 10.0,
        "WAPE": 7.5,
        "MEDIAN_AE": 75.0,
    }

    result = validation.build_eval_result(
        job_name="training-job",
        purpose="Evaluate model",
        pass_status=True,
        metrics_result=metrics_result,
        params=valid_params,
    )

    assert result == {
        "job_name": "training-job",
        "purpose": "Evaluate model",
        "status": "true",
        "hyperparameters": {
            "eta": 0.05,
            "max_depth": 8,
            "n_estimators": 600,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "metrics": metrics_result,
        "R2": 0.95,
        "MAPE": 8.0,
        "MAE": 100.0,
        "RMSE_RATIO": 1.5,
    }


def test_build_eval_result_uses_lowercase_false_status(
    valid_params,
):
    metrics_result = {
        "R2": 0.80,
        "MAPE": 12.0,
        "MAE": 100.0,
        "RMSE": 250.0,
        "RMSE_RATIO": 2.5,
    }

    result = validation.build_eval_result(
        job_name="training-job",
        purpose="Evaluate model",
        pass_status=False,
        metrics_result=metrics_result,
        params=valid_params,
    )

    assert result["status"] == "false"


@pytest.mark.parametrize(
    ("r2", "expected_status", "expected_annotation"),
    [
        (
            0.95,
            "true",
            "Model evaluation gate passed::",
        ),
        (
            0.80,
            "false",
            "Model evaluation gate failed::",
        ),
    ],
)
def test_validate_mlflow_metrics_runs_complete_validation(
    mocker,
    monkeypatch,
    tmp_path,
    mlflow_response,
    r2,
    expected_status,
    expected_annotation,
    capsys,
):
    job_name = "training-job"
    github_arguments = [
        "validate_model_metrics.py",
        "--iteration",
        "1",
        "--job_name",
        job_name,
        "--rg",
        "test-rg",
        "--workspace_name",
        "test-workspace",
        "--r2_min",
        "0.90",
        "--mape_max",
        "10",
        "--mae_max",
        "100000",
        "--rmse_ratio_max",
        "2.0",
        "--debug",
    ]

    for metric in mlflow_response["run"]["data"]["metrics"]:
        if metric["key"] == "R2":
            metric["value"] = r2

    monkeypatch.chdir(tmp_path)
    mocker.patch.object(sys, "argv", github_arguments)

    mock_tracking_uri = mocker.patch.object(
        validation,
        "get_tracking_uri",
        return_value="https://tracking.example",
    )
    mock_access_token = mocker.patch.object(
        validation,
        "get_access_token",
        return_value="access-token",
    )
    mock_run_json = mocker.patch.object(
        validation,
        "get_mlflow_run_json",
        return_value=mlflow_response,
    )

    validation.validate_mlflow_metrics()

    output_path = tmp_path / f"job_eval_model_{job_name}.json"

    assert output_path.is_file()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out

    assert result["job_name"] == job_name
    assert result["purpose"] == "Evaluate model"
    assert result["status"] == expected_status
    assert result["R2"] == r2

    assert "Validating iteration 1" in output
    assert f"Retrieving MLflow metrics for job: {job_name}" in output
    assert "Metrics:" in output
    assert "Params:" in output
    assert str(output_path.name) in output
    assert expected_annotation in output

    mock_tracking_uri.assert_called_once_with(
        job_name=job_name,
        resource_group="test-rg",
        workspace_name="test-workspace",
    )
    mock_access_token.assert_called_once_with()
    mock_run_json.assert_called_once_with(
        tracking_uri="https://tracking.example",
        job_name=job_name,
        token="access-token",
    )


def test_validate_mlflow_metrics_defaults_missing_optional_metrics(
    mocker,
    monkeypatch,
    tmp_path,
    valid_params,
):
    response = {
        "run": {
            "data": {
                "metrics": [
                    {"key": "R2", "value": 0.95},
                    {"key": "MAPE", "value": 8.0},
                    {"key": "MAE", "value": 100.0},
                    {"key": "RMSE", "value": 150.0},
                ],
                "params": [
                    {"key": key, "value": value}
                    for key, value in valid_params.items()
                ],
            }
        }
    }

    monkeypatch.chdir(tmp_path)
    mocker.patch.object(
        sys,
        "argv",
        [
            "validate_model_metrics.py",
            "--iteration",
            "1",
            "--job_name",
            "training-job",
            "--rg",
            "test-rg",
            "--workspace_name",
            "test-workspace",
            "--r2_min",
            "0.90",
            "--mape_max",
            "10",
            "--mae_max",
            "100000",
            "--rmse_ratio_max",
            "2.0",
        ],
    )

    mocker.patch.object(
        validation,
        "get_tracking_uri",
        return_value="https://tracking.example",
    )
    mocker.patch.object(
        validation,
        "get_access_token",
        return_value="access-token",
    )
    mocker.patch.object(
        validation,
        "get_mlflow_run_json",
        return_value=response,
    )

    validation.validate_mlflow_metrics()

    output_path = tmp_path / "job_eval_model_training-job.json"
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["metrics"]["BIAS"] is None
    assert result["metrics"]["WAPE"] is None
    assert result["metrics"]["MEDIAN_AE"] is None