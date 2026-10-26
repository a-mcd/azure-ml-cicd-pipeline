"""Shared pytest setup.

Stop Locust from gevent-patching ssl/socket on import. Patching after
urllib3/jwt/azure have loaded causes a RecursionError, and the tests
don't need gevent. CI sets the same variable in the workflow.

Put src on PYTHONPATH so tests work without installing the package.
pyproject.toml's pytest `pythonpath` setting covers imports in the test
process; this environment variable covers the CLI integration tests, which
run scripts such as select_winner.py in a separate Python process.
"""
import os
from pathlib import Path

os.environ.setdefault("LOCUST_SKIP_MONKEY_PATCH", "1")

SRC = Path(__file__).resolve().parents[1] / "src"
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in (str(SRC), os.environ.get("PYTHONPATH")) if path
)
