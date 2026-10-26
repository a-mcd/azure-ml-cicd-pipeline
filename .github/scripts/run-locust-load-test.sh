#!/usr/bin/env bash
# Run the Locust load test against a new real-time deployment and write
# the results to the job summary. Reports are written to locust/.
#
# Environment:
#   SCORE_URI               Scoring URI of the endpoint.
#   RT_ENDPOINT_NAME        Real-time endpoint name.
#   RT_DEPLOYMENT_NAME      Deployment under test.
#   AZUREML_DEPLOYMENT      Deployment the requests are routed to.
#   LOCUST_USERS            Number of simulated users.
#   LOCUST_SPAWN_RATE       Users started per second.
#   LOCUST_RUN_TIME         Test duration, for example 2m.
#   LOCUST_P95_MAX_MS       Maximum allowed p95 latency in milliseconds.
#   LOCUST_MAX_FAIL_RATIO   Maximum allowed share of failed requests.
#   LOCUST_STATS_FILE       JSON summary written by locust_load_test.py.
set -euo pipefail

echo "::group::Run Locust load test"
echo "Target URI:       $SCORE_URI"
echo "Deployment:       $RT_DEPLOYMENT_NAME"
echo "Users:            $LOCUST_USERS"
echo "Spawn rate:       $LOCUST_SPAWN_RATE users/second"
echo "Duration:         $LOCUST_RUN_TIME"
echo "p95 threshold:    ${LOCUST_P95_MAX_MS} ms"
echo "Failure threshold: $LOCUST_MAX_FAIL_RATIO"

echo "Running Locust against $SCORE_URI for deployment $RT_DEPLOYMENT_NAME"

mkdir -p locust
STATUS="unknown"
REQUEST_COUNT="unknown"
FAILURE_COUNT="unknown"
FAILURE_RATIO="unknown"
REQUESTS_PER_SECOND="unknown"
AVERAGE_RESPONSE_TIME="unknown"
P95_RESPONSE_TIME="unknown"
# Locust host is the full AzureML scoring URI; we call path ""
set +e
locust \
  --locustfile src/walmart_ml/pipeline/locust_load_test.py \
  --headless \
  --host "$SCORE_URI" \
  --users "$LOCUST_USERS" \
  --spawn-rate "$LOCUST_SPAWN_RATE" \
  --run-time "$LOCUST_RUN_TIME" \
  --csv locust/locust \
  --csv-full-history \
  --html locust/locust-report.html \
  --logfile locust/locust.log \
  --loglevel INFO
LOCUST_EXIT_CODE="$?"
set -e
echo "::endgroup::"

echo "::group::Load-test results"
if [[ -s "$LOCUST_STATS_FILE" ]] &&
   jq empty "$LOCUST_STATS_FILE" 2>/dev/null; then
  STATUS="$(jq -r '.status' "$LOCUST_STATS_FILE")"
  REQUEST_COUNT="$(jq -r '.requests' "$LOCUST_STATS_FILE")"
  FAILURE_COUNT="$(jq -r '.failures' "$LOCUST_STATS_FILE")"
  FAILURE_RATIO="$(jq -r '.failure_ratio' "$LOCUST_STATS_FILE")"
  REQUESTS_PER_SECOND="$(
    jq -r '.requests_per_second // "unknown"' \
      "$LOCUST_STATS_FILE"
  )"
  AVERAGE_RESPONSE_TIME="$(
    jq -r '.average_response_time_ms // "unknown"' \
      "$LOCUST_STATS_FILE"
  )"
  P95_RESPONSE_TIME="$(
    jq -r '.p95_response_time_ms // "unknown"' \
      "$LOCUST_STATS_FILE"
  )"

  jq . "$LOCUST_STATS_FILE"
else
  echo "::warning title=Missing Locust summary::$LOCUST_STATS_FILE was not created or is invalid"
fi

echo
echo "Status:                $STATUS"
echo "Exit code:             $LOCUST_EXIT_CODE"
echo "Requests:              $REQUEST_COUNT"
echo "Failures:              $FAILURE_COUNT"
echo "Failure ratio:         $FAILURE_RATIO"
echo "Requests per second:   $REQUESTS_PER_SECOND"
echo "Average response time: ${AVERAGE_RESPONSE_TIME} ms"
echo "p95 response time:     ${P95_RESPONSE_TIME} ms"
echo "::endgroup::"

{
  echo "## Real-time endpoint load test"
  echo
  echo "| Configuration | Value |"
  echo "|---|---:|"
  echo "| Endpoint | \`$RT_ENDPOINT_NAME\` |"
  echo "| Deployment | \`$RT_DEPLOYMENT_NAME\` |"
  echo "| Users | \`$LOCUST_USERS\` |"
  echo "| Spawn rate | \`${LOCUST_SPAWN_RATE} users/second\` |"
  echo "| Duration | \`$LOCUST_RUN_TIME\` |"
  echo "| Maximum p95 latency | \`${LOCUST_P95_MAX_MS} ms\` |"
  echo "| Maximum failure ratio | \`$LOCUST_MAX_FAIL_RATIO\` |"
  echo
  echo "### Results"
  echo
  echo "| Metric | Result |"
  echo "|---|---:|"
  echo "| Status | \`$STATUS\` |"
  echo "| Exit code | \`$LOCUST_EXIT_CODE\` |"
  echo "| Requests | \`$REQUEST_COUNT\` |"
  echo "| Failures | \`$FAILURE_COUNT\` |"
  echo "| Failure ratio | \`$FAILURE_RATIO\` |"
  echo "| Requests per second | \`$REQUESTS_PER_SECOND\` |"
  echo "| Average response time | \`${AVERAGE_RESPONSE_TIME} ms\` |"
  echo "| p95 response time | \`${P95_RESPONSE_TIME} ms\` |"
  echo
  echo "## Generated reports"
  echo
  echo "| Report | Artifact |"
  echo "|---|---|"
  echo "| Locust HTML, CSV, JSON and log reports | \`locust-load-test-${GITHUB_RUN_ID}\` |"
} >> "$GITHUB_STEP_SUMMARY"

echo "locust_exit_code=$LOCUST_EXIT_CODE" >> "$GITHUB_OUTPUT"

if [[ "$LOCUST_EXIT_CODE" -ne 0 ]]; then
  echo "::error title=Locust load test failed::Status=$STATUS, exit code=$LOCUST_EXIT_CODE, p95=${P95_RESPONSE_TIME} ms, failure ratio=$FAILURE_RATIO"
  exit "$LOCUST_EXIT_CODE"
fi
