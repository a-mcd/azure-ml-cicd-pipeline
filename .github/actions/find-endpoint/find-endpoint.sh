#!/usr/bin/env bash
# Find the endpoint whose name starts with ENDPOINT_PREFIX.
#
# Environment:
#   ENDPOINT_TYPE    "online" or "batch".
#   ENDPOINT_PREFIX  Endpoint name prefix.
#   RG               Resource group of the workspace.
#   WORKSPACE_NAME   Azure ML workspace.
#   ENV_VAR          Optional environment variable to save the name to.
set -euo pipefail

case "$ENDPOINT_TYPE" in
  online) LABEL="real-time" ;;
  batch) LABEL="batch" ;;
  *)
    echo "::error title=Invalid endpoint type::Expected 'online' or 'batch', got '$ENDPOINT_TYPE'"
    exit 1
    ;;
esac

echo "::group::Find matching $LABEL endpoints"
echo "Workspace:       $WORKSPACE_NAME"
echo "Resource group:  $RG"
echo "Endpoint prefix: $ENDPOINT_PREFIX"

MATCHING_ENDPOINTS="$(
  az ml "${ENDPOINT_TYPE}-endpoint" list \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "[?starts_with(name, '${ENDPOINT_PREFIX}')].name" \
    --output json
)"
MATCH_COUNT="$(jq 'length' <<< "$MATCHING_ENDPOINTS")"
ENDPOINT_NAME="$(jq -r 'sort | first // empty' <<< "$MATCHING_ENDPOINTS")"
echo "::endgroup::"

if [[ "$MATCH_COUNT" -eq 0 ]]; then
  echo "::error title=Endpoint not found::No $LABEL endpoint beginning with '$ENDPOINT_PREFIX' was found."
  exit 1
fi

if [[ "$MATCH_COUNT" -gt 1 ]]; then
  echo "::warning title=Multiple $LABEL endpoints::Selected '$ENDPOINT_NAME' from $MATCH_COUNT matches"
  echo "Matching endpoints:"
  jq -r '.[]' <<< "$MATCHING_ENDPOINTS"
fi

echo "Using $LABEL endpoint: $ENDPOINT_NAME"
echo "name=$ENDPOINT_NAME" >> "$GITHUB_OUTPUT"

if [[ -n "${ENV_VAR:-}" ]]; then
  echo "$ENV_VAR=$ENDPOINT_NAME" >> "$GITHUB_ENV"
fi
