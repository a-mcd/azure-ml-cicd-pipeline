#!/usr/bin/env python3
"""Print the latest date found in the Date column of a CSV file."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from walmart_ml.pipeline.github_actions import write_environment_variables


DATE_COLUMN = "Date"

ACCEPTED_DATE_FORMATS = (
    "%d-%m-%Y",
)

OUTPUT_DATE_FORMAT = "%d-%m-%Y"


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        argparse.Namespace: Parsed ``file`` and ``github_env`` options.
    """
    parser = argparse.ArgumentParser(
        description="Print the latest date in a CSV file."
    )
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the CSV file.",
    )

    parser.add_argument(
        '--github_env',
        '-g',
        action='store_true',
        help="Store date as environment variable"
    )

    return parser.parse_args()


def parse_date(value: str) -> datetime:
    """Parse a date string using the first accepted format that matches.

    Args:
        value: Date string from the CSV.

    Returns:
        datetime: The parsed date.

    Raises:
        ValueError: If the value matches none of ``ACCEPTED_DATE_FORMATS``.
    """
    for date_format in ACCEPTED_DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue

    raise ValueError(
        f"Date {value!r} does not match any accepted format: "
        f"{', '.join(ACCEPTED_DATE_FORMATS)}"
    )


def find_latest_date(csv_path: Path) -> datetime:
    """Return the latest non-empty date in the CSV's Date column.

    Args:
        csv_path: CSV file to read.

    Returns:
        datetime: The latest date.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the CSV has no Date column.
        ValueError: If a date cannot be parsed or no dates are present.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    latest: datetime | None = None

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames or DATE_COLUMN not in reader.fieldnames:
            available_columns = ", ".join(reader.fieldnames or [])

            raise KeyError(
                f"Column {DATE_COLUMN!r} was not found. "
                f"Available columns: {available_columns}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_value = (row.get(DATE_COLUMN) or "").strip()

            if not raw_value:
                continue

            try:
                parsed_date = parse_date(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{error} at CSV row {row_number}"
                ) from error

            if latest is None or parsed_date > latest:
                latest = parsed_date

    if latest is None:
        raise ValueError(
            f"No non-empty dates were found in column {DATE_COLUMN!r}"
        )

    return latest


def main() -> int:
    """Print the latest date in the CSV and the following week's date.

    With ``--github_env`` the dates are also written to ``$GITHUB_ENV`` as
    ``LATEST_DATE`` and ``MATERIALIZATION_END_DATE``.

    Returns:
        int: Process exit code, 0 on success.
    """
    args = parse_arguments()
    latest = find_latest_date(Path(args.file))

    latest_date = latest.strftime(OUTPUT_DATE_FORMAT)
    next_week = latest + timedelta(weeks=1)
    next_week_date = next_week.strftime(OUTPUT_DATE_FORMAT)
    print(f"Latest Date: {latest_date}")
    print(f"Materialization End Date: {next_week_date}")

    if args.github_env:
        write_environment_variables(
            {
                "LATEST_DATE": latest_date,
                "MATERIALIZATION_END_DATE": next_week_date,
            }
        )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
