"""Logging setup shared by the command-line entry points.

Library modules only create a module logger with ``logging.getLogger(__name__)``.
Each script's ``main()`` calls :func:`configure_logging` once, so every job in
Azure ML and GitHub Actions writes log lines in the same format.
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Send log records to stderr in the project's standard format.

    Logs go to stderr so that stdout carries only a script's real output
    (values, JSON results, GitHub Actions commands). Does nothing if the root
    logger already has handlers, for example when pytest is capturing logs.

    Args:
        level: Minimum level of the records to show.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
