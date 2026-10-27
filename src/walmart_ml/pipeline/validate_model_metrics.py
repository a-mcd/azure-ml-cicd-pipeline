"""Model evaluation gate for the staging workflow.

Fetches a training job's metrics and parameters from Azure ML's MLflow
endpoint, checks the metrics against thresholds passed on the command line,
and writes ``job_eval_model_<job>.json`` for the model-selection step. The
result is also printed, with GitHub Actions annotations for each check when
``--debug`` is set.
"""

from pathlib import Path
import argparse
import subprocess
import json
import logging
from typing import Any

from walmart_ml.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def mlflow_items_to_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a list of MLflow key/value items into a dictionary.

    For example, ``[{"key": "R2", "value": 0.91}, {"key": "MAPE", "value": 8.5}]``
    becomes ``{"R2": 0.91, "MAPE": 8.5}``. Items without a key are skipped.

    Args:
        items: Metric or param items from the MLflow REST API.

    Returns:
        dict[str, Any]: Item values by key.
    """
    return {
        item["key"]: item.get("value")
        for item in items
        if "key" in item
    }


def optional_float(value: Any) -> float | None:
    """Convert a metric value to a float, keeping missing values as ``None``.

    Args:
        value: Metric value from MLflow, which may be missing or empty.

    Returns:
        float | None: The value as a float, or ``None`` if it is missing.
    """
    if value in ("", None):
        return None

    return float(value)


def run_command(command: list[str]) -> str:
    """Run a command without a shell and return its stdout.

    Args:
        command: Program and arguments.

    Returns:
        str: The command's stdout, with surrounding whitespace removed.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero code.
    """
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def get_tracking_uri(job_name: str, resource_group: str, workspace_name: str) -> str:
    """Get the MLflow tracking endpoint from the Azure ML job.

    Args:
        job_name: Azure ML job name.
        resource_group: Resource group of the workspace.
        workspace_name: Azure ML workspace name.

    Returns:
        str: HTTPS base URL of the MLflow tracking server.
    """
    raw_uri = run_command([
        "az", "ml", "job", "show",
        "--name", job_name,
        "--resource-group", resource_group,
        "--workspace-name", workspace_name,
        "--query", "services.Tracking.endpoint",
        "-o", "tsv",
    ])

    return raw_uri.replace("azureml://", "https://").rstrip("?")


def get_access_token() -> str:
    """Get an Azure ML access token from the Azure CLI.

    Returns:
        str: Bearer token for ``https://ml.azure.com``.
    """
    return run_command([
        "az", "account", "get-access-token",
        "--resource", "https://ml.azure.com",
        "--query", "accessToken",
        "-o", "tsv",
    ])


def get_mlflow_run_json(tracking_uri: str, job_name: str, token: str) -> dict:
    """Query MLflow run data using the Azure ML tracking endpoint.

    Args:
        tracking_uri: Tracking server URL from :func:`get_tracking_uri`.
        job_name: Azure ML job name, which is also the MLflow run ID.
        token: Bearer token from :func:`get_access_token`.

    Returns:
        dict: The ``runs/get`` API response.
    """
    url = f"{tracking_uri}/api/2.0/mlflow/runs/get?run_id={job_name}"

    response = run_command([
        "curl",
        "-sS",
        "-H",
        f"Authorization: Bearer {token}",
        url,
    ])

    return json.loads(response)


def check_maximum(name: str, value: float, threshold: float, debug: bool = False) -> bool:
    """Check that a metric is less than or equal to its threshold.

    Args:
        name: Metric name, used in the annotation.
        value: Metric value.
        threshold: Maximum allowed value.
        debug: Whether to print a GitHub Actions notice or warning.

    Returns:
        bool: True if the metric passed.
    """
    passed = value <= threshold

    if debug:
        if passed:
            print(f"::notice title={name} passed::{name}={value}, maximum={threshold}")
        else:
            print(f"::warning title={name} failed::{name}={value}, maximum={threshold}")

    return passed


def check_minimum(name: str, value: float, threshold: float, debug: bool = False) -> bool:
    """Check that a metric is greater than or equal to its threshold.

    Args:
        name: Metric name, used in the annotation.
        value: Metric value.
        threshold: Minimum allowed value.
        debug: Whether to print a GitHub Actions notice or warning.

    Returns:
        bool: True if the metric passed.
    """
    passed = value >= threshold

    if debug:
        if passed:
            print(f"::notice title={name} passed::{name}={value}, minimum={threshold}")
        else:
            print(f"::warning title={name} failed::{name}={value}, minimum={threshold}")

    return passed


def validate_model_metrics(
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    debug: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Validate model metrics against thresholds.

    Args:
        metrics: MLflow metrics. Must include R2, MAPE, MAE and RMSE.
        thresholds: ``R2_MIN``, ``MAPE_MAX``, ``MAE_MAX`` and ``RMSE_RATIO_MAX``.
        debug: Whether to print an annotation for each check.

    Returns:
        tuple[bool, dict[str, Any]]: Whether every check passed, and the
        metric values that were checked, including ``RMSE_RATIO``.

    Raises:
        ValueError: If a required metric is missing or MAE is zero.
    """

    missing = [
        name for name in ["R2", "MAPE", "MAE", "RMSE"]
        if name not in metrics or metrics[name] in ("", None)
    ]

    if missing:
        raise ValueError(f"Missing required metrics: {missing}")

    r2 = float(metrics["R2"])
    mape = float(metrics["MAPE"])
    mae = float(metrics["MAE"])
    rmse = float(metrics["RMSE"])

    if mae == 0:
        raise ValueError("MAE is zero, cannot calculate RMSE_RATIO")

    rmse_ratio = round(rmse / mae, 6)

    checks = [
        check_minimum("R2", r2, thresholds["R2_MIN"], debug=debug),
        check_maximum("MAPE", mape, thresholds["MAPE_MAX"], debug=debug),
        check_maximum("MAE", mae, thresholds["MAE_MAX"], debug=debug),
        check_maximum("RMSE_RATIO", rmse_ratio, thresholds["RMSE_RATIO_MAX"], debug=debug),
    ]

    metrics_result = {
        "R2": r2,
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse,
        "RMSE_RATIO": rmse_ratio,
        "BIAS": optional_float(metrics.get("Bias")),
        "WAPE": optional_float(metrics.get("WAPE")),
        "MEDIAN_AE": optional_float(metrics.get("Median_AE")),
    }

    return all(checks), metrics_result


def build_eval_result(
    job_name: str,
    purpose: str,
    pass_status: bool,
    metrics_result: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Build the final JSON result for later model selection.

    Args:
        job_name: Azure ML job name.
        purpose: Short description of why the job was evaluated.
        pass_status: Whether the model passed the evaluation gate.
        metrics_result: Checked metrics from :func:`validate_model_metrics`.
        params: MLflow params, including the XGBoost hyperparameters.

    Returns:
        dict[str, Any]: Result with status, hyperparameters and metrics. The
        ranking metrics (WAPE, RMSE, BIAS and MAPE) are also copied to the top
        level so the workflow can sort candidates with ``jq``.
    """
    return {
        "job_name": job_name,
        "purpose": purpose,
        "status": str(pass_status).lower(),
        "hyperparameters": {
            "eta": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "n_estimators": int(params["n_estimators"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "reg_alpha": float(params["reg_alpha"]),
            "reg_lambda": float(params["reg_lambda"]),
        },
        "metrics": metrics_result,
        "R2": metrics_result["R2"],
        "MAPE": metrics_result["MAPE"],
        "MAE": metrics_result["MAE"],
        "RMSE": metrics_result["RMSE"],
        "RMSE_RATIO": metrics_result["RMSE_RATIO"],
        "WAPE": metrics_result["WAPE"],
        "BIAS": metrics_result["BIAS"],
    }


def parse_args() -> argparse.Namespace:
    """Parse the evaluation gate's command-line arguments.

    Returns:
        argparse.Namespace: Job details, metric thresholds and the debug flag.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True, help='Iteration')
    parser.add_argument("--job_name", type=str, required=True, help='Azure job name')
    parser.add_argument("--rg", type=str, required=True, help='Azure resource group')
    parser.add_argument("--workspace_name", type=str, required=True, help='Azure workspace name')
    parser.add_argument("--r2_min", type=float, required=True, help='R2 min threshold')
    parser.add_argument("--mape_max", type=int, required=True, help='MAPE max threshold')
    parser.add_argument("--mae_max", type=int, required=True, help='MAE max threshold')
    parser.add_argument("--rmse_ratio_max", type=float, required=True, help='RMSE ratio max threshold')
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed metric threshold checks.",
    )
    return parser.parse_args()


def fetch_run_data(
    job_name: str,
    resource_group: str,
    workspace_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch a training job's MLflow metrics and params.

    Args:
        job_name: Azure ML job name.
        resource_group: Resource group of the workspace.
        workspace_name: Azure ML workspace name.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The run's metrics and params.
    """
    tracking_uri = get_tracking_uri(
        job_name=job_name,
        resource_group=resource_group,
        workspace_name=workspace_name,
    )

    token = get_access_token()

    run_json = get_mlflow_run_json(
        tracking_uri=tracking_uri,
        job_name=job_name,
        token=token,
    )

    run_data = run_json["run"]["data"]
    metrics = mlflow_items_to_dict(run_data.get("metrics", []))
    params = mlflow_items_to_dict(run_data.get("params", []))

    LOGGER.info("Metrics:\n%s", json.dumps(metrics, indent=2))
    LOGGER.info("Params:\n%s", json.dumps(params, indent=2))

    return metrics, params


def format_gate_summary(
    job_name: str,
    iteration: int,
    metrics_result: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    """Describe each checked metric against its threshold on one line.

    Args:
        job_name: Azure ML job name.
        iteration: Iteration number from the workflow.
        metrics_result: Checked metrics from :func:`validate_model_metrics`.
        thresholds: Thresholds the metrics were checked against.

    Returns:
        str: Summary for the gate's final log line.
    """
    return (
        f"job={job_name}, "
        f"iteration={iteration}, "
        f"R2={metrics_result['R2']:.4f} >= {thresholds['R2_MIN']:.4f}, "
        f"MAPE={metrics_result['MAPE']:.4f} <= {thresholds['MAPE_MAX']}, "
        f"MAE={metrics_result['MAE']:.2f} <= {thresholds['MAE_MAX']}, "
        f"RMSE_RATIO={metrics_result['RMSE_RATIO']:.4f} <= {thresholds['RMSE_RATIO_MAX']:.4f}"
    )


def validate_mlflow_metrics() -> None:
    """Run the evaluation gate from the command line.

    Parses the arguments, fetches the job's metrics, checks them and writes the
    result file. The final line printed says whether the gate passed.
    """
    configure_logging()
    args = parse_args()

    thresholds = {
        "R2_MIN": args.r2_min,
        "MAPE_MAX": args.mape_max,
        "MAE_MAX": args.mae_max,
        "RMSE_RATIO_MAX": args.rmse_ratio_max,
    }

    LOGGER.info("Validating iteration %s", args.iteration)
    LOGGER.info("Retrieving MLflow metrics for job: %s", args.job_name)

    metrics, params = fetch_run_data(args.job_name, args.rg, args.workspace_name)

    pass_status, metrics_result = validate_model_metrics(metrics, thresholds, debug=args.debug)

    result = build_eval_result(
        job_name=args.job_name,
        purpose="Evaluate model",
        pass_status=pass_status,
        metrics_result=metrics_result,
        params=params,
    )

    output_path = Path(f"job_eval_model_{args.job_name}.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))

    print(output_path)

    summary = format_gate_summary(args.job_name, args.iteration, metrics_result, thresholds)

    if pass_status:
        print(f"Model evaluation gate passed::{summary}")
    else:
        print(f"Model evaluation gate failed::{summary}")


if __name__ == "__main__":
    validate_mlflow_metrics()
