import os
import json
from pathlib import Path
from locust import HttpUser, task, between, events

# Load a sample payload once
PAYLOAD_PATH = os.getenv("LOCUST_PAYLOAD_PATH", "tests/test.json")
with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
    REQUEST_BODY = json.load(f)


class WalmartRealtimeUser(HttpUser):
    """
    Simulated client calling the Azure ML real-time endpoint.
    """
    wait_time = between(0.1, 1.0)  # short wait to simulate bursty load
    headers = None

    def on_start(self):
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

    @task
    def score_request(self):
        """
        Call the scoring endpoint with a realistic request body.
        Using name='score' so we can inspect stats easily.
        """
        with self.client.post(
            "",  # empty path since --host will be full scoring_uri
            name="score",
            headers=self.headers,
            json=REQUEST_BODY,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Bad status: {resp.status_code}, body: {resp.text[:200]}")
            else:
                # If you want, you can do basic schema checks here
                resp.success()


# ---- Gating logic: fail CI if too slow / too many errors ----

@events.test_stop.add_listener
def assert_performance(environment, **_kwargs):
    """Write load-test statistics and fail when thresholds are exceeded."""
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

    print(f"Load-test statistics written to: {stats_file}")
    print(json.dumps(results, indent=2))

    if passed:
        print("Load test passed all performance thresholds.")
        environment.process_exit_code = 0
    else:
        for reason in failure_reasons:
            print(f"Load-test failure: {reason}")

        environment.process_exit_code = 1
