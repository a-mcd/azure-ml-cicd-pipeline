"""Unit tests for the Locust Azure ML load test."""

import json
from types import SimpleNamespace

import pytest

from walmart_ml.pipeline_scripts import locust_load_test as load_test


def create_user(mocker):
    """Create a WalmartRealtimeUser without starting Locust."""
    user = object.__new__(load_test.WalmartRealtimeUser)
    user.client = mocker.MagicMock()
    user.headers = None
    return user


def create_environment(
    mocker,
    *,
    num_requests,
    p95=0.0,
    fail_ratio=0.0,
):
    """Create a mock Locust environment with aggregated statistics."""
    total_stats = mocker.Mock()
    total_stats.num_requests = num_requests
    total_stats.num_failures = int(num_requests * fail_ratio)
    total_stats.fail_ratio = fail_ratio
    total_stats.total_rps = 12.5
    total_stats.total_fail_per_sec = 0.25
    total_stats.avg_response_time = 250.0
    total_stats.median_response_time = 200.0
    total_stats.min_response_time = 100.0
    total_stats.max_response_time = 1000.0
    total_stats.avg_content_length = 512.0
    total_stats.get_response_time_percentile.return_value = p95

    environment = SimpleNamespace(
        stats=SimpleNamespace(total=total_stats),
        process_exit_code=None,
    )

    return environment, total_stats


@pytest.fixture(autouse=True)
def locust_stats_file(monkeypatch, tmp_path):
    """Write each test's load-test summary into its temporary directory."""
    stats_file = tmp_path / "locust-summary.json"
    monkeypatch.setenv("LOCUST_STATS_FILE", str(stats_file))
    return stats_file


def test_request_body_is_loaded_from_json():
    assert isinstance(load_test.REQUEST_BODY, (dict, list))


def test_user_has_expected_wait_time():
    assert callable(load_test.WalmartRealtimeUser.wait_time)


def test_on_start_creates_required_headers(
    mocker,
    monkeypatch,
):
    user = create_user(mocker)

    monkeypatch.setenv("AZUREML_TOKEN", "test-token")
    monkeypatch.delenv("AZUREML_DEPLOYMENT", raising=False)

    user.on_start()

    assert user.headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }


def test_on_start_adds_deployment_header(
    mocker,
    monkeypatch,
):
    user = create_user(mocker)

    monkeypatch.setenv("AZUREML_TOKEN", "test-token")
    monkeypatch.setenv(
        "AZUREML_DEPLOYMENT",
        "rt-deployment-20260804",
    )

    user.on_start()

    assert user.headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
        (
            "azureml-model-deployment"
        ): "rt-deployment-20260804",
    }


def test_on_start_ignores_empty_deployment_name(
    mocker,
    monkeypatch,
):
    user = create_user(mocker)

    monkeypatch.setenv("AZUREML_TOKEN", "test-token")
    monkeypatch.setenv("AZUREML_DEPLOYMENT", "")

    user.on_start()

    assert "azureml-model-deployment" not in user.headers


def test_on_start_requires_token(
    mocker,
    monkeypatch,
):
    user = create_user(mocker)

    monkeypatch.delenv("AZUREML_TOKEN", raising=False)
    monkeypatch.delenv("AZUREML_DEPLOYMENT", raising=False)

    with pytest.raises(KeyError, match="AZUREML_TOKEN"):
        user.on_start()


def test_score_request_marks_success_for_200_response(
    mocker,
):
    user = create_user(mocker)
    user.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }

    response = mocker.Mock()
    response.status_code = 200

    context_manager = mocker.MagicMock()
    context_manager.__enter__.return_value = response
    user.client.post.return_value = context_manager

    user.score_request()

    user.client.post.assert_called_once_with(
        "",
        name="score",
        headers=user.headers,
        json=load_test.REQUEST_BODY,
        catch_response=True,
    )
    response.success.assert_called_once_with()
    response.failure.assert_not_called()


def test_score_request_marks_failure_for_non_200_response(
    mocker,
):
    user = create_user(mocker)
    user.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }

    response = mocker.Mock()
    response.status_code = 500
    response.text = "Internal server error"

    context_manager = mocker.MagicMock()
    context_manager.__enter__.return_value = response
    user.client.post.return_value = context_manager

    user.score_request()

    response.failure.assert_called_once_with(
        "Bad status: 500, body: Internal server error"
    )
    response.success.assert_not_called()


def test_score_request_truncates_failure_body_to_200_characters(
    mocker,
):
    user = create_user(mocker)
    user.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }

    response = mocker.Mock()
    response.status_code = 400
    response.text = "x" * 300

    context_manager = mocker.MagicMock()
    context_manager.__enter__.return_value = response
    user.client.post.return_value = context_manager

    user.score_request()

    expected_message = (
        "Bad status: 400, body:"
        f" {'x' * 200}"
    )
    response.failure.assert_called_once_with(expected_message)
    response.success.assert_not_called()


def test_assert_performance_fails_when_no_requests_recorded(
    mocker,
    capsys,
    locust_stats_file,
):
    environment, total_stats = create_environment(
        mocker,
        num_requests=0,
    )

    load_test.assert_performance(environment)

    assert environment.process_exit_code == 1
    assert "Load-test failure: No requests were recorded" in capsys.readouterr().out
    results = json.loads(locust_stats_file.read_text(encoding="utf-8"))
    assert results["status"] == "failed"
    assert results["requests"] == 0
    assert results["p95_response_time_ms"] is None
    assert results["failure_reasons"] == ["No requests were recorded"]
    total_stats.get_response_time_percentile.assert_not_called()


def test_assert_performance_passes_with_default_thresholds(
    mocker,
    monkeypatch,
    capsys,
    locust_stats_file,
):
    monkeypatch.delenv("LOCUST_P95_MAX_MS", raising=False)
    monkeypatch.delenv(
        "LOCUST_MAX_FAIL_RATIO",
        raising=False,
    )

    environment, total_stats = create_environment(
        mocker,
        num_requests=100,
        p95=1000.0,
        fail_ratio=0.005,
    )

    load_test.assert_performance(environment)

    output = capsys.readouterr().out

    assert environment.process_exit_code == 0
    assert "Load test passed all performance thresholds." in output

    results = json.loads(locust_stats_file.read_text(encoding="utf-8"))
    assert results["status"] == "passed"
    assert results["requests"] == 100
    assert results["failures"] == 0
    assert results["failure_ratio"] == 0.005
    assert results["p95_response_time_ms"] == 1000.0
    assert results["thresholds"] == {
        "p95_response_time_ms": 1500.0,
        "failure_ratio": 0.01,
    }
    assert results["failure_reasons"] == []

    total_stats.get_response_time_percentile.assert_called_once_with(
        0.95
    )


def test_assert_performance_accepts_values_at_threshold(
    mocker,
    monkeypatch,
):
    monkeypatch.setenv("LOCUST_P95_MAX_MS", "1000")
    monkeypatch.setenv("LOCUST_MAX_FAIL_RATIO", "0.05")

    environment, _ = create_environment(
        mocker,
        num_requests=100,
        p95=1000.0,
        fail_ratio=0.05,
    )

    load_test.assert_performance(environment)

    assert environment.process_exit_code == 0


def test_assert_performance_fails_when_p95_exceeds_threshold(
    mocker,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("LOCUST_P95_MAX_MS", "1000")
    monkeypatch.setenv("LOCUST_MAX_FAIL_RATIO", "0.05")

    environment, _ = create_environment(
        mocker,
        num_requests=100,
        p95=1001.0,
        fail_ratio=0.01,
    )

    load_test.assert_performance(environment)

    output = capsys.readouterr().out

    assert environment.process_exit_code == 1
    assert "Load-test failure: p95 latency 1001.0 ms exceeded 1000.0 ms" in output
    assert "Load-test failure: Failure ratio" not in output


def test_assert_performance_fails_when_failure_ratio_exceeds_threshold(
    mocker,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("LOCUST_P95_MAX_MS", "1000")
    monkeypatch.setenv("LOCUST_MAX_FAIL_RATIO", "0.05")

    environment, _ = create_environment(
        mocker,
        num_requests=100,
        p95=900.0,
        fail_ratio=0.051,
    )

    load_test.assert_performance(environment)

    output = capsys.readouterr().out

    assert environment.process_exit_code == 1
    assert "Load-test failure: Failure ratio 0.0510 exceeded 0.0500" in output
    assert "Load-test failure: p95 latency" not in output


def test_assert_performance_fails_when_both_thresholds_exceeded(
    mocker,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("LOCUST_P95_MAX_MS", "1000")
    monkeypatch.setenv("LOCUST_MAX_FAIL_RATIO", "0.05")

    environment, _ = create_environment(
        mocker,
        num_requests=100,
        p95=1500.0,
        fail_ratio=0.10,
    )

    load_test.assert_performance(environment)

    output = capsys.readouterr().out

    assert environment.process_exit_code == 1
    assert "Load-test failure: p95 latency 1500.0 ms exceeded 1000.0 ms" in output
    assert "Load-test failure: Failure ratio 0.1000 exceeded 0.0500" in output


def test_assert_performance_ignores_additional_keyword_arguments(
    mocker,
):
    environment, _ = create_environment(
        mocker,
        num_requests=10,
        p95=500.0,
        fail_ratio=0.0,
    )

    load_test.assert_performance(
        environment,
        runner="ignored",
        extra_value=True,
    )

    assert environment.process_exit_code == 0