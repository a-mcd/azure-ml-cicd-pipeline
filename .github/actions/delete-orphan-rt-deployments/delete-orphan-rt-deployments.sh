#!/usr/bin/env bash
# Delete real-time deployments that receive no traffic.
#
# Only deployments in the Succeeded or Failed state are deleted; any still
# being created or updated are skipped and listed in the summary.
#
# Environment:
#   RT_ENDPOINT_NAME  Real-time endpoint name.
#   RG                Resource group of the workspace.
#   WORKSPACE_NAME    Azure ML workspace.
set -euo pipefail

echo "::group::Query real-time endpoint"

TRAFFIC_JSON="$(
  az ml online-endpoint show \
    --name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query traffic \
    --output json
)"

DEPLOYMENTS_JSON="$(
  az ml online-deployment list \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query '[].{
      name:name,
      state:provisioning_state,
      instanceType:instance_type
    }' \
    --output json
)"

DEPLOYMENT_COUNT="$(jq 'length' <<< "$DEPLOYMENTS_JSON")"
TRAFFIC_TOTAL="$(jq '[.[]] | add // 0' <<< "$TRAFFIC_JSON")"

echo "Endpoint:         $RT_ENDPOINT_NAME"
echo "Resource group:   $RG"
echo "Workspace:        $WORKSPACE_NAME"
echo "Deployment count: $DEPLOYMENT_COUNT"
echo "Traffic total:    ${TRAFFIC_TOTAL}%"
echo
echo "Current deployments:"

if [[ "$DEPLOYMENT_COUNT" -eq 0 ]]; then
  echo "No real-time deployments exist."
else
  jq -nr \
    --argjson deployments "$DEPLOYMENTS_JSON" \
    --argjson traffic "$TRAFFIC_JSON" \
    '
      ["DEPLOYMENT", "STATE", "INSTANCE TYPE", "TRAFFIC"],
      (
        $deployments[]
        | [
            .name,
            .state,
            .instanceType,
            "\($traffic[.name] // 0)%"
          ]
      )
      | @tsv
    ' |
    column -t -s $'\t'
fi

echo "::endgroup::"

# Explicitly declare the variables as arrays. This prevents
# unbound-variable errors when no deployments match.
NO_TRAFFIC_DEPLOYMENTS=()
DELETED_DEPLOYMENTS=()
SKIPPED_DEPLOYMENTS=()

mapfile -t NO_TRAFFIC_DEPLOYMENTS < <(
  jq -nr \
    --argjson deployments "$DEPLOYMENTS_JSON" \
    --argjson traffic "$TRAFFIC_JSON" \
    '
      $deployments[]
      | select(($traffic[.name] // 0) == 0)
      | select(
          .state == "Succeeded" or
          .state == "Failed"
        )
      | .name
    '
)

mapfile -t SKIPPED_DEPLOYMENTS < <(
  jq -nr \
    --argjson deployments "$DEPLOYMENTS_JSON" \
    --argjson traffic "$TRAFFIC_JSON" \
    '
      $deployments[]
      | select(($traffic[.name] // 0) == 0)
      | select(
          .state != "Succeeded" and
          .state != "Failed"
        )
      | "\(.name) (\(.state))"
    '
)

NO_TRAFFIC_COUNT="${#NO_TRAFFIC_DEPLOYMENTS[@]}"
SKIPPED_COUNT="${#SKIPPED_DEPLOYMENTS[@]}"

echo "::group::Real-time deployment cleanup plan"
echo "Eligible for deletion: $NO_TRAFFIC_COUNT"
echo "Skipped:               $SKIPPED_COUNT"

if [[ "$NO_TRAFFIC_COUNT" -eq 0 ]]; then
  echo "No zero-traffic deployments are eligible for deletion."
else
  echo
  echo "Deployments to delete:"
  printf '%s\n' "${NO_TRAFFIC_DEPLOYMENTS[@]}"
fi

if [[ "$SKIPPED_COUNT" -gt 0 ]]; then
  echo
  echo "Deployments skipped because of their state:"
  printf '%s\n' "${SKIPPED_DEPLOYMENTS[@]}"
fi

echo "::endgroup::"

for DEPLOYMENT_NAME in "${NO_TRAFFIC_DEPLOYMENTS[@]}"; do
  echo "::group::Delete deployment $DEPLOYMENT_NAME"

  az ml online-deployment delete \
    --name "$DEPLOYMENT_NAME" \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --yes

  DELETED_DEPLOYMENTS+=("$DEPLOYMENT_NAME")

  echo "Deleted deployment: $DEPLOYMENT_NAME"
  echo "::endgroup::"
done

DELETED_COUNT="${#DELETED_DEPLOYMENTS[@]}"

REMAINING_DEPLOYMENTS_JSON="$(
  az ml online-deployment list \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query '[].name' \
    --output json
)"

REMAINING_COUNT="$(
  jq 'length' <<< "$REMAINING_DEPLOYMENTS_JSON"
)"

echo "::group::Real-time deployment cleanup result"
echo "Deleted deployments:   $DELETED_COUNT"
echo "Skipped deployments:   $SKIPPED_COUNT"
echo "Remaining deployments: $REMAINING_COUNT"

if [[ "$REMAINING_COUNT" -gt 0 ]]; then
  echo
  echo "Remaining deployments:"
  jq -r '.[]' <<< "$REMAINING_DEPLOYMENTS_JSON"
fi

echo "::endgroup::"

{
  echo "## Real-time deployment cleanup"
  echo
  echo "| Result | Count |"
  echo "|---|---:|"
  echo "| Deleted | \`$DELETED_COUNT\` |"
  echo "| Skipped | \`$SKIPPED_COUNT\` |"
  echo "| Remaining | \`$REMAINING_COUNT\` |"
  echo
  echo "### Deleted deployments"
  echo

  if [[ "$DELETED_COUNT" -eq 0 ]]; then
    echo "No real-time deployments were deleted."
  else
    for DEPLOYMENT_NAME in "${DELETED_DEPLOYMENTS[@]}"; do
      echo "- \`$DEPLOYMENT_NAME\`"
    done
  fi

  if [[ "$SKIPPED_COUNT" -gt 0 ]]; then
    echo
    echo "### Skipped deployments"
    echo

    for DEPLOYMENT in "${SKIPPED_DEPLOYMENTS[@]}"; do
      echo "- \`$DEPLOYMENT\`"
    done
  fi
} >> "$GITHUB_STEP_SUMMARY"