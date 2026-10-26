#!/usr/bin/env bash
# Send 100% of a real-time endpoint's traffic to one deployment.
#
# Environment:
#   RT_ENDPOINT_NAME    Real-time endpoint name.
#   RT_DEPLOYMENT_NAME  Deployment that should receive all traffic.
#   RG                  Resource group of the workspace.
#   WORKSPACE_NAME      Azure ML workspace.
set -euo pipefail

print_traffic() {
  # Print a traffic JSON object ({"deployment": percent}) as a table.
  if [[ "$(jq 'length' <<< "$1")" -eq 0 ]]; then
    echo "No deployment currently receives endpoint traffic."
  else
    jq -r '
      ["DEPLOYMENT", "TRAFFIC"],
      (to_entries[] | [.key, "\(.value)%"])
      | @tsv
    ' <<< "$1" |
      column -t -s $'\t'
  fi
}

echo "::group::Current traffic allocation"
PREVIOUS_TRAFFIC="$(
  az ml online-endpoint show \
    --name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query traffic \
    --output json |
  jq --compact-output '.'
)"
print_traffic "$PREVIOUS_TRAFFIC"
echo "::endgroup::"

echo "::group::Update traffic"
echo "Endpoint:       $RT_ENDPOINT_NAME"
echo "New deployment: $RT_DEPLOYMENT_NAME"
echo "Target traffic: 100%"
echo "Resource group: $RG"
echo "Workspace:      $WORKSPACE_NAME"

RT_TRAFFIC_STATE="$(
  az ml online-endpoint update \
    --name "$RT_ENDPOINT_NAME" \
    --traffic "$RT_DEPLOYMENT_NAME=100" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" |
  jq -r '.provisioning_state'
)"

if [[ "$RT_TRAFFIC_STATE" != "Succeeded" ]]; then
  echo "::error title=Traffic update failed::Could not update '$RT_DEPLOYMENT_NAME' traffic (provisioning state: $RT_TRAFFIC_STATE)"
  exit 1
fi
echo "Traffic for deployment '$RT_DEPLOYMENT_NAME' updated successfully."
echo "::endgroup::"

echo "::group::Verify final traffic allocation"
ENDPOINT_JSON="$(
  az ml online-endpoint show \
    --name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --output json
)"

FINAL_STATE="$(
  jq -r '.provisioning_state // empty' <<< "$ENDPOINT_JSON"
)"

FINAL_TRAFFIC_JSON="$(
  jq --compact-output '.traffic // {}' <<< "$ENDPOINT_JSON"
)"

TARGET_TRAFFIC="$(
  jq --raw-output \
    --arg deployment "$RT_DEPLOYMENT_NAME" \
    '.traffic[$deployment] // 0' \
    <<< "$ENDPOINT_JSON"
)"

TOTAL_TRAFFIC="$(
  jq '
    [.traffic[]?] | add // 0
  ' <<< "$ENDPOINT_JSON"
)"

echo "Update response state: $RT_TRAFFIC_STATE"
echo "Final endpoint state:  $FINAL_STATE"
echo "Target deployment:     $RT_DEPLOYMENT_NAME"
echo "Target traffic:        ${TARGET_TRAFFIC}%"
echo "Total traffic:         ${TOTAL_TRAFFIC}%"
echo
echo "Final traffic allocation:"
print_traffic "$FINAL_TRAFFIC_JSON"

echo "::endgroup::"

if [[ "$FINAL_STATE" != "Succeeded" ]]; then
  echo "::error title=Endpoint update failed::Endpoint provisioning state is '$FINAL_STATE'"
  exit 1
fi

if [[ "$TARGET_TRAFFIC" -ne 100 ]]; then
  echo "::error title=Incorrect traffic allocation::$RT_DEPLOYMENT_NAME received ${TARGET_TRAFFIC}% instead of 100%"
  exit 1
fi

if [[ "$TOTAL_TRAFFIC" -ne 100 ]]; then
  echo "::error title=Invalid total traffic::Endpoint traffic totals ${TOTAL_TRAFFIC}% instead of 100%"
  exit 1
fi

{
  echo "## Real-time endpoint traffic update"
  echo
  echo "| Property | Value |"
  echo "|---|---|"
  echo "| Endpoint | \`$RT_ENDPOINT_NAME\` |"
  echo "| Active deployment | \`$RT_DEPLOYMENT_NAME\` |"
  echo "| Traffic allocation | \`${TARGET_TRAFFIC}%\` |"
  echo "| Endpoint state | \`$FINAL_STATE\` |"
  echo "| Resource group | \`$RG\` |"
  echo "| Workspace | \`$WORKSPACE_NAME\` |"
  echo
  echo "### Traffic change"
  echo
  echo "| Allocation | Value |"
  echo "|---|---|"
  echo "| Previous | \`$PREVIOUS_TRAFFIC\` |"
  echo "| Current | \`$FINAL_TRAFFIC_JSON\` |"
} >> "$GITHUB_STEP_SUMMARY"

{
  echo "previous-traffic=$PREVIOUS_TRAFFIC"
  echo "current-traffic=$FINAL_TRAFFIC_JSON"
} >> "$GITHUB_OUTPUT"
