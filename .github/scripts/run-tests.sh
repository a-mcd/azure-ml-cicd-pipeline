#!/usr/bin/env bash
# Run the unit and integration tests with coverage, and summarise the
# results in the job summary. Reports are written to reports/.
#
# Environment:
#   TEST_COVERAGE_THRESHOLD  Minimum total coverage, in percent.
set -euo pipefail

mkdir -p reports

echo "::group::Run pytest tests"
set +e

LOCUST_SKIP_MONKEY_PATCH=1 \
pytest tests \
  --maxfail=1 \
  --junitxml=reports/tests.xml \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=xml:reports/coverage.xml \
  --cov-fail-under="$TEST_COVERAGE_THRESHOLD" \
  -q \
  2>&1 | tee reports/pytest-output.txt

PYTEST_EXIT_CODE=$?
set -e
echo "::endgroup::"

echo "::group::Test results summary"

TEST_SUMMARY="$(
  grep -E \
    '[0-9]+ (passed|failed|error|errors|skipped|deselected)' \
    reports/pytest-output.txt |
  tail -n 1 ||
  true
)"

if [[ -z "$TEST_SUMMARY" ]]; then
  TEST_SUMMARY="Pytest did not produce a recognizable test summary"
fi

COVERAGE_LINE="$(
  grep -E \
    'TOTAL[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+%' \
    reports/pytest-output.txt |
  tail -n 1 ||
  true
)"

COVERAGE_PERCENT="$(
  awk '{print $4}' <<< "$COVERAGE_LINE" |
  tr -d '%' ||
  true
)"

echo "Pytest exit code: $PYTEST_EXIT_CODE"

if [[ "$PYTEST_EXIT_CODE" -eq 0 ]]; then
  TEST_STATUS="Passed"
else
  TEST_STATUS="Failed"
  echo "::error title=Tests failed::$TEST_SUMMARY; Pytest exit code $PYTEST_EXIT_CODE"
fi

if [[ -z "$COVERAGE_PERCENT" ]]; then
  COVERAGE_STATUS="Unknown"
  echo "::warning title=Test coverage unknown::Could not extract coverage percentage from Pytest output"
else
  COVERAGE_STATUS="Passed"
fi

echo "Pytest exit code: $PYTEST_EXIT_CODE"
echo "Test result:      $TEST_STATUS"
echo "Test summary:     $TEST_SUMMARY"
echo "Coverage:         ${COVERAGE_PERCENT:-unknown}%"
echo "Coverage result:  $COVERAGE_STATUS"

echo "::endgroup::"

echo "::group::Generated reports"
ls -la reports
{
  echo "## Unit and integration tests"
  echo
  echo "| Check | Result |"
  echo "|---|---|"
  echo "| Tests | $TEST_STATUS |"
  echo "| Coverage | $COVERAGE_STATUS |"
  echo
  echo "### Results"
  echo
  echo "- Test summary: \`$TEST_SUMMARY\`"
  echo "- Coverage: \`${COVERAGE_PERCENT:-unknown}%\`"
  echo "- Required coverage: \`${TEST_COVERAGE_THRESHOLD}%\`"
  echo "- Pytest exit code: \`$PYTEST_EXIT_CODE\`"
  echo
  echo "### Generated reports"
  echo
  echo "- \`tests.xml\`: JUnit test results"
  echo "- \`coverage.xml\`: XML coverage report"
  echo "- \`pytest-output.txt\`: Complete Pytest output"
} >> "$GITHUB_STEP_SUMMARY"

exit $PYTEST_EXIT_CODE
