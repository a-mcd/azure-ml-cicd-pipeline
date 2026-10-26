"""Locust load test for the Azure ML real-time endpoint.

Run by the staging workflow against a new deployment. Each simulated user
posts the sample request in ``LOCUST_PAYLOAD_PATH``. When the test stops,
:func:`assert_performance` writes a JSON summary and sets a non-zero exit code
if the 95th-percentile latency or the failure ratio is over its threshold.

Environment variables:
    AZUREML_TOKEN: Bearer token for the endpoint (required).
    AZUREML_DEPLOYMENT: Deployment to route requests to (optional).
    LOCUST_PAYLOAD_PATH: Request body JSON file.
    LOCUST_P95_MAX_MS: Maximum allowed p95 latency in milliseconds.
    LOCUST_MAX_FAIL_RATIO: Maximum allowed share of failed requests.
    LOCUST_STATS_FILE: Where to write the JSON summary.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any

from locust import HttpUser, task, between, events
from locust.env import Environment

LOGGER = logging.getLogger(__name__)

DEFAULT_PAYLOAD_PATH = "tests/test_data.json"


def load_request_body() -> Any:
    """Read the sample request body sent by every simulated user.

    Returns:
        Any: Parsed JSON from ``LOCUST_PAYLOAD_PATH``, or from
        ``tests/test_data.json`` when the variable is not set.
    """
    payload_path = Path(os.getenv("LOCUST_PAYLOAD_PATH", DEFAULT_PAYLOAD_PATH))
    with payload_path.open("r", encoding="utf-8") as payload_file:
        return json.load(payload_file)


class WalmartRealtimeUser(HttpUser):
    """Simulated client calling the Azure ML real-time endpoint.

    Attributes:
        wait_time: Pause between requests, kept short to simulate bursty load.
        headers: Request headers, set in :meth:`on_start`.
        request_body: JSON body for each request, set in :meth:`on_start`.
    """
    wait_time = between(0.1, 1.0)  # short wait to simulate bursty load
    headers: dict[str, str] | None = None
    request_body: Any = None

    def on_start(self) -> None:
        """Build the auth headers and load the request body once per simulated user."""
        token = os.environ["AZUREML_TOKEN"]
        deployment_name = os.environ.get("AZUREML_DEPLOYMENT", "")

        # Headers required by Azure ML endpoint
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if deployment_name:
            # Route specifically to the new deployment
            self.headers["azureml-model-deployment"] = deployment_name

        self.request_body = load_request_body()

    @task
    def score_request(self) -> None:
        """Call the scoring endpoint with the sample request body.

        The request is named ``score`` so its statistics are easy to find.
        """
        with self.client.post(
            "",  # empty path since --host will be full scoring_uri
            name="score",
            headers=self.headers,
            json=self.request_body,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Bad status: {resp.status_code}, body: {resp.text[:200]}")
            else:
                # If you want, you can do basic schema checks here
                resp.success()


# ---- Gating logic: fail CI if too slow / too many errors ----

@events.test_stop.add_listener
def assert_performance(environment: Environment, **_kwargs: Any) -> None:
    """Write load-test statistics and fail when thresholds are exceeded.

    Registered as a Locust ``quitting`` event listener.

    Args:
        environment: Locust environment holding the run's statistics.
        **_kwargs: Other event arguments, unused.
    """
    stats = environment.stats.total

    max_p95_ms = float(
        os.getenv("LOCUST_P95_MAX_MS", "1500")
    )
    max_fail_ratio = float(
        os.getenv("LOCUST_MAX_FAIL_RATIO", "0.01")
    )
    stats_file = Path(
        os.getenv(
            "LOCUST_STATS_FILE",
            "locust/locust-summary.json",
        )
    )

    stats_file.parent.mkdir(parents=True, exist_ok=True)

    p95 = (
        stats.get_response_time_percentile(0.95)
        if stats.num_requests > 0
        else None
    )

    failure_reasons = []

    if stats.num_requests == 0:
        failure_reasons.append("No requests were recorded")

    if p95 is not None and p95 > max_p95_ms:
        failure_reasons.append(
            f"p95 latency {p95} ms exceeded {max_p95_ms} ms"
        )

    if stats.fail_ratio > max_fail_ratio:
        failure_reasons.append(
            f"Failure ratio {stats.fail_ratio:.4f} exceeded "
            f"{max_fail_ratio:.4f}"
        )

    passed = not failure_reasons

    results = {
        "status": "passed" if passed else "failed",
        "requests": stats.num_requests,
        "failures": stats.num_failures,
        "failure_ratio": stats.fail_ratio,
        "requests_per_second": stats.total_rps,
        "failures_per_second": stats.total_fail_per_sec,
        "average_response_time_ms": stats.avg_response_time,
        "median_response_time_ms": stats.median_response_time,
        "minimum_response_time_ms": stats.min_response_time,
        "maximum_response_time_ms": stats.max_response_time,
        "p95_response_time_ms": p95,
        "average_content_size_bytes": stats.avg_content_length,
        "thresholds": {
            "p95_response_time_ms": max_p95_ms,
            "failure_ratio": max_fail_ratio,
        },
        "failure_reasons": failure_reasons,
    }

    with stats_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    LOGGER.info("Load-test statistics written to: %s", stats_file)
    print(json.dumps(results, indent=2))

    if passed:
        print("Load test passed all performance thresholds.")
        environment.process_exit_code = 0
    else:
        for reason in failure_reasons:
            print(f"Load-test failure: {reason}")

        environment.process_exit_code = 1
