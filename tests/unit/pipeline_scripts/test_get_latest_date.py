"""Unit tests for the latest CSV date helper."""

import argparse
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from src.pipeline_scripts import get_latest_date_csv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parse_arguments_reads_required_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_latest_date.py",
            "--file",
            "data/Walmart_Sales.csv",
        ],
    )

    result = get_latest_date_csv.parse_arguments()

    assert isinstance(result, argparse.Namespace)
    assert result.file == "data/Walmart_Sales.csv"
    assert result.github_env is False


@pytest.mark.parametrize("flag", ["--github_env", "-g"])
def test_parse_arguments_enables_github_environment_output(
    monkeypatch,
    flag,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_latest_date_csv.py",
            "--file",
            "data/Walmart_Sales.csv",
            flag,
        ],
    )

    result = get_latest_date_csv.parse_arguments()

    assert result.file == "data/Walmart_Sales.csv"
    assert result.github_env is True


@pytest.mark.parametrize("flag", ["--file", "-f"])
def test_parse_arguments_accepts_both_file_flags(
    monkeypatch,
    flag,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_latest_date_csv.py",
            flag,
            "data/input.csv",
        ],
    )

    result = get_latest_date_csv.parse_arguments()

    assert result.file == "data/input.csv"


def test_parse_arguments_requires_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["get_latest_date_csv.py"],
    )

    with pytest.raises(SystemExit) as error:
        get_latest_date_csv.parse_arguments()

    assert error.value.code == 2


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("01-01-2024", datetime(2024, 1, 1)),
        ("29-02-2024", datetime(2024, 2, 29)),
        ("31-12-2024", datetime(2024, 12, 31)),
    ],
)
def test_parse_date_accepts_supported_dates(value, expected):
    assert get_latest_date_csv.parse_date(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-01",
        "01/01/2024",
        "32-01-2024",
        "29-02-2023",
        "",
        "not-a-date",
    ],
)
def test_parse_date_rejects_unsupported_or_invalid_dates(value):
    with pytest.raises(
        ValueError,
        match=(
            rf"Date {value!r} does not match any accepted format: "
            r"%d-%m-%Y"
        ),
    ):
        get_latest_date_csv.parse_date(value)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def configure_arguments(
    mocker,
    get_latest_date_module,
    csv_path,
    *,
    github_env=False,
):
    """Configure the parsed command-line arguments for a main test."""
    return mocker.patch.object(
        get_latest_date_module,
        "parse_arguments",
        return_value=SimpleNamespace(
            file=str(csv_path),
            github_env=github_env,
        ),
    )


def test_main_prints_latest_date_and_next_week(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date,Weekly_Sales\n"
        "1,05-01-2024,1000\n"
        "2,19-01-2024,2000\n"
        "3,12-01-2024,3000\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )
    mock_write_environment = mocker.patch.object(
        get_latest_date_csv,
        "write_environment_variables",
    )

    result = get_latest_date_csv.main()

    assert result == 0
    mock_write_environment.assert_not_called()

    captured = capsys.readouterr()

    assert captured.out == (
        "Latest Date: 19-01-2024\n"
        "Materialisation End Date: 26-01-2024\n"
    )
    assert captured.err == ""


def test_main_writes_dates_to_github_environment(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1,26-01-2024\n"
        "2,02-02-2024\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
        github_env=True,
    )
    mock_write_environment = mocker.patch.object(
        get_latest_date_csv,
        "write_environment_variables",
    )

    result = get_latest_date_csv.main()

    assert result == 0

    mock_write_environment.assert_called_once_with({
        "LATEST_DATE": "02-02-2024",
        "MATERIALISATION_END_DATE": "09-02-2024",
    })

    captured = capsys.readouterr()

    assert "Latest Date: 02-02-2024" in captured.out
    assert "Materialisation End Date: 09-02-2024" in captured.out


def test_main_ignores_empty_date_values(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1,\n"
        "2,  \n"
        "3,12-01-2024\n"
        "4,\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    result = get_latest_date_csv.main()

    assert result == 0

    captured = capsys.readouterr()
    assert "Latest Date: 12-01-2024" in captured.out
    assert "Materialisation End Date: 19-01-2024" in captured.out


def test_main_strips_date_whitespace(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1, 05-01-2024 \n"
        "2, 12-01-2024 \n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    assert get_latest_date_csv.main() == 0

    captured = capsys.readouterr()
    assert "Latest Date: 12-01-2024" in captured.out


def test_main_reads_csv_with_utf8_byte_order_mark(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "\ufeffStore,Date\n"
        "1,05-01-2024\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    assert get_latest_date_csv.main() == 0

    captured = capsys.readouterr()
    assert "Latest Date: 05-01-2024" in captured.out


def test_main_raises_when_csv_does_not_exist(
    mocker,
    tmp_path,
):
    csv_path = tmp_path / "missing.csv"

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    with pytest.raises(
        FileNotFoundError,
        match=rf"CSV file not found: {csv_path}",
    ):
        get_latest_date_csv.main()


@pytest.mark.parametrize(
    ("content", "available_columns"),
    [
        ("", ""),
        (
            "Store,Weekly_Sales\n"
            "1,1000\n",
            "Store, Weekly_Sales",
        ),
    ],
)
def test_main_raises_when_date_column_is_missing(
    mocker,
    tmp_path,
    content,
    available_columns,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(content, encoding="utf-8")

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    with pytest.raises(
        KeyError,
        match=(
            r"Column 'Date' was not found\. "
            rf"Available columns: {available_columns}"
        ),
    ):
        get_latest_date_csv.main()


def test_main_raises_when_all_dates_are_empty(
    mocker,
    tmp_path,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1,\n"
        "2,  \n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match=r"No non-empty dates were found in column 'Date'",
    ):
        get_latest_date_csv.main()


def test_main_reports_csv_row_number_for_invalid_date(
    mocker,
    tmp_path,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1,05-01-2024\n"
        "2,invalid-date\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Date 'invalid-date' does not match any accepted format: "
            r"%d-%m-%Y at CSV row 3"
        ),
    ):
        get_latest_date_csv.main()


def test_main_handles_year_boundary(
    mocker,
    tmp_path,
    capsys,
):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "Store,Date\n"
        "1,29-12-2023\n",
        encoding="utf-8",
    )

    configure_arguments(
        mocker,
        get_latest_date_csv,
        csv_path,
    )

    assert get_latest_date_csv.main() == 0

    captured = capsys.readouterr()

    assert "Latest Date: 29-12-2023" in captured.out
    assert "Materialisation End Date: 05-01-2024" in captured.out