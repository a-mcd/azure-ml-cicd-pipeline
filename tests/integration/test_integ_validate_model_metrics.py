"""Integration tests for the model-metric validation CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "src"
    / "walmart_ml"
    / "pipeline_scripts"
    / "validate_model_metrics.py"
)


@pytest.fixture
def fake_command_directory(tmp_path: Path) -> Path:
    """Create fake az and curl command-line programs."""
    command_directory = tmp_path / "bin"
    command_directory.mkdir()

    az_script = command_directory / "az"
    curl_script = command_directory / "curl"

    az_script.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path


arguments = sys.argv[1:]
log_path = Path(os.environ["FAKE_COMMAND_LOG"])

with log_path.open("a", encoding="utf-8") as log:
    log.write(
        json.dumps({{"command": "az", "args": arguments}})
        + "\\n"
    )

if arguments[:3] == ["ml", "job", "show"]:
    print("azureml://tracking.example?")
elif arguments[:2] == ["account", "get-access-token"]:
    print("integration-access-token")
else:
    print(
        f"Unexpected az arguments: {{arguments}}",
        file=sys.stderr,
    )
    sys.exit(2)
""",
        encoding="utf-8",
    )

    curl_script.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path


arguments = sys.argv[1:]
log_path = Path(os.environ["FAKE_COMMAND_LOG"])
response_path = Path(
    os.environ["FAKE_MLFLOW_RESPONSE_PATH"]
)

with log_path.open("a", encoding="utf-8") as log:
    log.write(
        json.dumps({{"command": "curl", "args": arguments}})
        + "\\n"
    )

print(response_path.read_text(encoding="utf-8"))
""",
        encoding="utf-8",
    )

    az_script.chmod(0o755)
    curl_script.chmod(0o755)

    return command_directory


def create_mlflow_response(r2: float) -> dict:
    """Create a representative Azure MLflow API response."""
    return {
        "run": {
            "data": {
                "metrics": [
                    {"key": "R2", "value": r2},
                    {"key": "MAPE", "value": 8.0},
                    {"key": "MAE", "value": 100.0},
                    {"key": "RMSE", "value": 150.0},
                    {"key": "Bias", "value": 10.0},
                    {"key": "WAPE", "value": 7.5},
                    {"key": "Median_AE", "value": 75.0},
                ],
                "params": [
                    {
                        "key": "learning_rate",
                        "value": "0.05",
                    },
                    {
                        "key": "max_depth",
                        "value": "8",
                    },
                    {
                        "key": "n_estimators",
                        "value": "600",
                    },
                    {
                        "key": "subsample",
                        "value": "0.8",
                    },
                    {
                        "key": "colsample_bytree",
                        "value": "0.8",
                    },
                    {
                        "key": "reg_alpha",
                        "value": "0.0",
                    },
                    {
                        "key": "reg_lambda",
                        "value": "1.0",
                    },
                ],
            }
        }
    }


@pytest.mark.parametrize(
    (
        "r2",
        "expected_status",
        "expected_annotation",
    ),
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
def test_validate_model_metrics_cli_end_to_end(
    tmp_path: Path,
    fake_command_directory: Path,
    r2: float,
    expected_status: str,
    expected_annotation: str,
):
    """Run the complete metric-validation CLI workflow."""
    job_name = "integration-training-job"

    response_path = tmp_path / "mlflow-response.json"
    command_log_path = tmp_path / "commands.jsonl"

    response_path.write_text(
        json.dumps(create_mlflow_response(r2)),
        encoding="utf-8",
    )
    command_log_path.touch()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["FAKE_COMMAND_LOG"] = str(
        command_log_path
    )
    environment["FAKE_MLFLOW_RESPONSE_PATH"] = str(
        response_path
    )
    environment["PATH"] = (
        str(fake_command_directory)
        + os.pathsep
        + environment["PATH"]
    )

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--iteration",
        "1",
        "--job_name",
        job_name,
        "--rg",
        "integration-rg",
        "--workspace_name",
        "integration-workspace",
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

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    diagnostics = (
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert result.returncode == 0, diagnostics
    assert result.stderr == ""

    assert "Validating iteration 1" in result.stdout
    assert (
        f"Retrieving MLflow metrics for job: {job_name}"
        in result.stdout
    )
    assert "Metrics:" in result.stdout
    assert "Params:" in result.stdout
    assert expected_annotation in result.stdout

    output_path = (
        tmp_path / f"job_eval_model_{job_name}.json"
    )

    assert output_path.is_file()

    evaluation = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert evaluation["job_name"] == job_name
    assert evaluation["purpose"] == "Evaluate model"
    assert evaluation["status"] == expected_status

    assert evaluation["hyperparameters"] == {
        "eta": 0.05,
        "max_depth": 8,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }

    assert evaluation["metrics"] == {
        "R2": r2,
        "MAPE": 8.0,
        "MAE": 100.0,
        "RMSE": 150.0,
        "RMSE_RATIO": 1.5,
        "BIAS": 10.0,
        "WAPE": 7.5,
        "MEDIAN_AE": 75.0,
    }

    assert evaluation["R2"] == r2
    assert evaluation["MAPE"] == 8.0
    assert evaluation["MAE"] == 100.0
    assert evaluation["RMSE_RATIO"] == 1.5

    command_records = [
        json.loads(line)
        for line in command_log_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(command_records) == 3

    job_show = command_records[0]

    assert job_show["command"] == "az"
    assert job_show["args"] == [
        "ml",
        "job",
        "show",
        "--name",
        job_name,
        "--resource-group",
        "integration-rg",
        "--workspace-name",
        "integration-workspace",
        "--query",
        "services.Tracking.endpoint",
        "-o",
        "tsv",
    ]

    access_token = command_records[1]

    assert access_token["command"] == "az"
    assert access_token["args"] == [
        "account",
        "get-access-token",
        "--resource",
        "https://ml.azure.com",
        "--query",
        "accessToken",
        "-o",
        "tsv",
    ]

    curl_request = command_records[2]

    assert curl_request["command"] == "curl"
    assert "-sS" in curl_request["args"]
    assert (
        "Authorization: Bearer integration-access-token"
        in curl_request["args"]
    )
    assert curl_request["args"][-1] == (
        "https://tracking.example/api/2.0/mlflow/"
        f"runs/get?run_id={job_name}"
    )