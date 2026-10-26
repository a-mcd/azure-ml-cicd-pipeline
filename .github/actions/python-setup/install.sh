#!/usr/bin/env bash
# Install the walmart_ml package in editable mode with the requested extras.
#
# Environment:
#   EXTRAS        Comma-separated pyproject extras (may be empty).
#   VERIFY_SPARK  "true" to check Java and the distutils shim PySpark needs.
set -euo pipefail

if [[ -n "${EXTRAS:-}" ]]; then
  PACKAGE=".[${EXTRAS}]"
else
  PACKAGE="."
fi

echo "::group::Install $PACKAGE"
python -m pip install --upgrade pip
python -m pip install -e "$PACKAGE"
echo "::endgroup::"

echo "::group::Verify Python environment"
python --version
python -m pip --version
python -m pip check

if [[ "${VERIFY_SPARK:-false}" == "true" ]]; then
  java -version 2>&1
  echo "JAVA_HOME=${JAVA_HOME:-}"
  # PySpark 3.5 imports distutils, which Python 3.12 only has through setuptools.
  python -c "from distutils.version import LooseVersion; print('distutils compatibility is available')"
fi
echo "::endgroup::"
