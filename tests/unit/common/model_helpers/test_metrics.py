import numpy as np
import pytest

from src.common.model_helpers.metrics import (
    calculate_regression_metrics,
    mape,
)


def test_mape_calculates_expected_percentage_error():
    # Purpose: Check that MAPE calculates the expected mean absolute percentage error
    # and returns the result as a float.
    y_true = [100, 200, 400]
    y_pred = [110, 180, 360]

    result = mape(y_true, y_pred)

    expected = np.mean([
        abs((100 - 110) / 100),
        abs((200 - 180) / 200),
        abs((400 - 360) / 400),
    ]) * 100

    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_mape_handles_zero_actual_values_without_division_error():
    # Purpose: Check that MAPE handles zero actual values without a division error
    # and still returns the result as a float.
    y_true = [0, 100]
    y_pred = [10, 90]

    result = mape(y_true, y_pred)

    expected = np.mean([
        abs((0 - 10) / 1e-9),
        abs((100 - 90) / 100),
    ]) * 100

    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_calculate_regression_metrics_logs_correct_metric_values(mocker, capsys):
    # Purpose: Check that calculate_regression_metrics logs the correct metric values,
    # prints a readable summary, and returns None.
    mock_log_metric = mocker.patch("src.common.model_helpers.metrics.mlflow.log_metric")

    y_test = np.array([100.0, 200.0, 300.0])
    preds = np.array([110.0, 190.0, 330.0])

    result = calculate_regression_metrics(y_test, preds)

    logged_metrics = {
        call.args[0]: call.args[1]
        for call in mock_log_metric.call_args_list
    }

    assert logged_metrics["MAE"] == pytest.approx(16.6666666667)
    assert logged_metrics["RMSE"] == pytest.approx(19.1485421551)
    assert logged_metrics["MAPE"] == pytest.approx(8.3333333333)
    assert logged_metrics["R2"] == pytest.approx(0.945)
    assert logged_metrics["Median_AE"] == pytest.approx(10.0)
    assert logged_metrics["Bias"] == pytest.approx(10.0)
    assert logged_metrics["WAPE"] == pytest.approx(8.3333333333)

    captured = capsys.readouterr()

    assert "MAE" in captured.out
    assert "RMSE" in captured.out
    assert "MAPE" in captured.out
    assert "R2" in captured.out
    assert "Median AE" in captured.out
    assert "BIAS" in captured.out
    assert "WAPE" in captured.out

    assert result is None