from pathlib import Path
import argparse
import subprocess
import json


def mlflow_items_to_dict(items):
    """
    Convert MLflow API items like:
    [
      {"key": "R2", "value": 0.91},
      {"key": "MAPE", "value": 8.5}
    ]

    into:
    {
      "R2": 0.91,
      "MAPE": 8.5
    }
    """
    return {
        item["key"]: item.get("value")
        for item in items
        if "key" in item
    }


def run_command(command: list[str]) -> str:
    """
    Run a shell command safely and return stdout.
    """
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def get_tracking_uri(job_name: str, resource_group: str, workspace_name: str) -> str:
    """
    Get the MLflow tracking endpoint from the Azure ML job.
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
    """
    Get Azure ML access token.
    """
    return run_command([
        "az", "account", "get-access-token",
        "--resource", "https://ml.azure.com",
        "--query", "accessToken",
        "-o", "tsv",
    ])


def get_mlflow_run_json(tracking_uri: str, job_name: str, token: str) -> dict:
    """
    Query MLflow run data using the Azure ML tracking endpoint.
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


def check_maximum(name, value, threshold, debug=False):
    """
    Check metric must be <= threshold.
    """
    passed = value <= threshold

    if debug:
        if passed:
            print(f"::notice title={name} passed::{name}={value}, maximum={threshold}")
        else:
            print(f"::warning title={name} failed::{name}={value}, maximum={threshold}")

    return passed


def check_minimum(name, value, threshold, debug=False):
    """
    Check metric must be >= threshold.
    """
    passed = value >= threshold

    if debug:
        if passed:
            print(f"::notice title={name} passed::{name}={value}, minimum={threshold}")
        else:
            print(f"::warning title={name} failed::{name}={value}, minimum={threshold}")

    return passed


def validate_model_metrics(metrics: dict, thresholds: dict, debug: bool = False) -> tuple[bool, dict]:
    """
    Validate model metrics against thresholds.
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
        "BIAS": metrics.get("Bias"),
        "WAPE": metrics.get("WAPE"),
        "MEDIAN_AE": metrics.get("Median_AE"),
    }

    return all(checks), metrics_result


def build_eval_result(
    job_name,
    purpose,
    pass_status,
    metrics_result,
    params,
):
    """
    Build final JSON result for later model selection.
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
        "RMSE_RATIO": metrics_result["RMSE_RATIO"],
    }


def validate_mlflow_metrics():
    """Main callable function for GitHub Actions."""

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
    args = parser.parse_args()

    iteration = args.iteration
    job_name = args.job_name
    resource_group = args.rg
    workspace_name = args.workspace_name

    thresholds = {
        "R2_MIN": args.r2_min,
        "MAPE_MAX": args.mape_max,
        "MAE_MAX": args.mae_max,
        "RMSE_RATIO_MAX": args.rmse_ratio_max,
    }

    print(f"Validating iteration {iteration}")
    print(f"Retrieving MLflow metrics for job: {job_name}")

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

    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    print("Params:")
    print(json.dumps(params, indent=2))

    pass_status, metrics_result = validate_model_metrics(metrics, thresholds, debug=args.debug)

    result = build_eval_result(
        job_name=job_name,
        purpose="Evaluate model",
        pass_status=pass_status,
        metrics_result=metrics_result,
        params=params,
    )

    output_path = Path(f"job_eval_model_{job_name}.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))

    print(output_path)

    summary = (
        f"job={job_name}, "
        f"iteration={iteration}, "
        f"R2={metrics_result['R2']:.4f} >= {thresholds['R2_MIN']:.4f}, "
        f"MAPE={metrics_result['MAPE']:.4f} <= {thresholds['MAPE_MAX']}, "
        f"MAE={metrics_result['MAE']:.2f} <= {thresholds['MAE_MAX']}, "
        f"RMSE_RATIO={metrics_result['RMSE_RATIO']:.4f} <= {thresholds['RMSE_RATIO_MAX']:.4f}"
    )

    if pass_status:
        print(f"Model evaluation gate passed::{summary}")
    else:
        print(f"Model evaluation gate failed::{summary}")



if __name__ == "__main__":
    validate_mlflow_metrics()
