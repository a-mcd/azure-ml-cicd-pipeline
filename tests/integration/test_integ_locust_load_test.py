"""Integration test for the Locust load-test script."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCUST_FILE = (
    PROJECT_ROOT
    / "src"
    / "walmart_ml"
    / "pipeline"
    / "locust_load_test.py"
)


class RecordingRequestHandler(BaseHTTPRequestHandler):
    """Record Locust requests and return successful responses."""

    received_requests: list[dict] = []

    def do_POST(self):
        """Record a POST request and return HTTP 200."""
        content_length = int(
            self.headers.get("Content-Length", "0")
        )
        raw_body = self.rfile.read(content_length)

        self.__class__.received_requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(raw_body),
            }
        )

        response_body = json.dumps(
            {
                "predictions": [
                    {
                        "Store": 1,
                        "Feature_Date": "02-11-2012",
                        "Prediction_Date": "09-11-2012",
                        "Predicted_Weekly_Sales": 123456.78,
                    }
                ]
            }
        ).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Content-Length",
            str(len(response_body)),
        )
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, _format, *_args):
        """Suppress HTTP server request logging."""


@pytest.fixture
def scoring_server():
    """Start a local scoring endpoint."""
    RecordingRequestHandler.received_requests = []

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        RecordingRequestHandler,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )
    server_thread.start()

    host = (
        f"http://127.0.0.1:"
        f"{server.server_address[1]}"
    )

    yield host

    server.shutdown()
    server.server_close()
    server_thread.join(timeout=5)


def test_locust_sends_requests_and_passes_performance_gate(
    tmp_path,
    scoring_server,
):
    """Run the actual Locust test against a local endpoint."""
    payload_path = tmp_path / "request.json"

    expected_payload = {
        "input_data": [
            {
                "Store": 1,
                "Date": "02-11-2012",
            }
        ]
    }

    payload_path.write_text(
        json.dumps(expected_payload),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    environment["LOCUST_PAYLOAD_PATH"] = str(payload_path)
    environment["AZUREML_TOKEN"] = "integration-test-token"
    environment["AZUREML_DEPLOYMENT"] = (
        "integration-test-deployment"
    )
    environment["LOCUST_P95_MAX_MS"] = "5000"
    environment["LOCUST_MAX_FAIL_RATIO"] = "0.0"

    # This subprocess runs a real Locust test, so gevent
    # monkey-patching must remain enabled.
    environment.pop("LOCUST_SKIP_MONKEY_PATCH", None)

    command = [
        sys.executable,
        "-m",
        "locust",
        "--locustfile",
        str(LOCUST_FILE),
        "--headless",
        "--users",
        "1",
        "--spawn-rate",
        "1",
        "--run-time",
        "2s",
        "--host",
        scoring_server,
        "--only-summary",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    diagnostic_output = (
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert result.returncode == 0, diagnostic_output

    assert RecordingRequestHandler.received_requests, (
        diagnostic_output
    )

    request = RecordingRequestHandler.received_requests[0]

    assert request["path"] == "/"
    assert request["body"] == expected_payload

    headers = {
        name.lower(): value
        for name, value in request["headers"].items()
    }

    assert headers["content-type"] == "application/json"
    assert (
        headers["authorization"]
        == "Bearer integration-test-token"
    )
    assert (
        headers["azureml-model-deployment"]
        == "integration-test-deployment"
    )

    combined_output = result.stdout + result.stderr

    assert "Load-test statistics written to:" in combined_output
    assert '"status": "passed"' in combined_output
    assert '"p95_response_time_ms":' in combined_output
    assert '"failure_ratio":' in combined_output
    assert '"failure_reasons": []' in combined_output
    assert "Load-test failure:" not in combined_output
    assert "Load test passed all performance thresholds." in combined_output