#!/usr/bin/env bash
# Smoke-test a real-time deployment with a request for store 1.
#
# Environment:
#   RT_ENDPOINT_NAME    Real-time endpoint name.
#   RT_DEPLOYMENT_NAME  Deployment to call.
#   REQUEST_DATE        Feature date to request (DD-MM-YYYY).
#   RG                  Resource group of the workspace.
#   WORKSPACE_NAME      Azure ML workspace.
set -euo pipefail

REQUEST_FILE="${RUNNER_TEMP}/walmart-endpoint-request.json"
RESPONSE_FILE="${RUNNER_TEMP}/walmart-endpoint-response.json"
RAW_RESPONSE_FILE="${RUNNER_TEMP}/walmart-endpoint-raw-response.json"

echo "::group::Prepare endpoint test"
if [[ -z "${REQUEST_DATE:-}" ]]; then
  echo "::error::No request date was given (LATEST_PROCESSED_DATE is not set)"
  exit 1
fi

jq -n \
    --arg date "$REQUEST_DATE" \
    '{
        input_data: [
            {
                Store: 1,
                Date: $date
            }
        ]
    }' > "$REQUEST_FILE"

echo "Request:"
jq . "$REQUEST_FILE"
echo "::endgroup::"

echo "::group::Invoke real-time deployment"
echo "Calling deployment: $RT_DEPLOYMENT_NAME"

if ! az ml online-endpoint invoke \
  --name "$RT_ENDPOINT_NAME" \
  --deployment-name "$RT_DEPLOYMENT_NAME" \
  --resource-group "$RG" \
  --workspace-name "$WORKSPACE_NAME" \
  --request-file "$REQUEST_FILE" \
  --output json > "$RAW_RESPONSE_FILE"; then

  echo "::error title=Endpoint invocation failed::Azure CLI could not invoke the deployment."

  az ml online-deployment get-logs \
    --name "$RT_DEPLOYMENT_NAME" \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --lines 200 || true

  exit 1
fi
echo "::endgroup::"

echo "::group::Process endpoint response"
if [[ ! -s "$RAW_RESPONSE_FILE" ]]; then
  echo "::error title=Empty endpoint response::The endpoint returned no content."
  exit 1
fi

jq -r 'fromjson' "$RAW_RESPONSE_FILE" > "$RESPONSE_FILE"

echo "Endpoint response:"
jq . "$RESPONSE_FILE"

# 1. Check for API/Model errors
if jq -e 'has("error")' "$RESPONSE_FILE" >/dev/null; then
  ERROR_TYPE="$(jq -r '.error' "$RESPONSE_FILE")"
  ERROR_MESSAGE="$(jq -r '.message // "No error message returned"' "$RESPONSE_FILE")"

  echo "::error title=Endpoint scoring failed::$ERROR_TYPE: $ERROR_MESSAGE"
  exit 1
fi

# 2. Validate response structure and field presence
jq -e \
  --arg request_date "$REQUEST_DATE" \
  '
    (.predictions | type == "array") and
    (.predictions | length > 0) and
    (.predictions[0].Store == 1) and
    (.predictions[0].Feature_Date == $request_date) and
    (.predictions[0] | has("Prediction_Date")) and
    (.predictions[0] | has("Predicted_Weekly_Sales")) and
    (.predictions[0].Predicted_Weekly_Sales != null)
  ' "$RESPONSE_FILE" >/dev/null || {
    echo "::error title=Invalid endpoint response::The response structure did not match the expected format."
    exit 1
  }

# 3. Extract variables safely
STORE="$(jq -r '.predictions[0].Store' "$RESPONSE_FILE")"
FEATURE_DATE="$(jq -r '.predictions[0].Feature_Date' "$RESPONSE_FILE")"
PREDICTION_DATE="$(jq -r '.predictions[0].Prediction_Date' "$RESPONSE_FILE")"
PREDICTED_SALES="$(jq -r '.predictions[0].Predicted_Weekly_Sales' "$RESPONSE_FILE")"
echo "::endgroup::"

echo "Endpoint test succeeded."
echo "Store: $STORE"
echo "Feature date: $FEATURE_DATE"
echo "Prediction date: $PREDICTION_DATE"
echo "Predicted weekly sales: $PREDICTED_SALES"

{
  echo "store=$STORE"
  echo "feature-date=$FEATURE_DATE"
  echo "prediction-date=$PREDICTION_DATE"
  echo "predicted-sales=$PREDICTED_SALES"
} >> "$GITHUB_OUTPUT"

{
  echo "### Real-time smoke test"
  echo
  echo "| Property | Value |"
  echo "|---|---|"
  echo "| Deployment | \`$RT_DEPLOYMENT_NAME\` |"
  echo "| Status | Passed |"
  echo "| Store | \`$STORE\` |"
  echo "| Feature date | \`$FEATURE_DATE\` |"
  echo "| Prediction date | \`$PREDICTION_DATE\` |"
  echo "| Predicted weekly sales | \`$PREDICTED_SALES\` |"
} >> "$GITHUB_STEP_SUMMARY"
