#!/usr/bin/env bash
# Delete batch deployments other than the endpoint's default.
#
# Deployments still being created or updated are skipped.
#
# Environment:
#   BATCH_ENDPOINT_NAME  Batch endpoint name.
#   RG                   Resource group of the workspace.
#   WORKSPACE_NAME       Azure ML workspace.
set -euo pipefail
DEFAULT_DEPLOYMENT=$(az ml batch-endpoint show -g "$RG" -w "$WORKSPACE_NAME" -n "$BATCH_ENDPOINT_NAME" --query "defaults.deployment_name" -o tsv)

if [[ -z "$DEFAULT_DEPLOYMENT" ||
  "$DEFAULT_DEPLOYMENT" == "null" ]]; then
  echo "::error title=Missing default batch deployment::Cleanup stopped to avoid deleting every batch deployment"
  exit 1
fi



BATCH_DEPLOYMENTS_JSON="$(
  az ml batch-deployment list \
    --endpoint-name "$BATCH_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query '[].{
      name:name,
      state:provisioning_state
    }' \
    --output json
)"
BATCH_DEPLOYMENT_COUNT="$(
  jq 'length' <<< "$BATCH_DEPLOYMENTS_JSON"
)"

DEPLOYMENTS_TO_DEL=$(az ml batch-deployment list -g "$RG" -w "$WORKSPACE_NAME" -e "$BATCH_ENDPOINT_NAME" --query "[?name!='$DEFAULT_DEPLOYMENT'].[name, provisioning_state]" -o tsv)

echo "Endpoint:         $BATCH_ENDPOINT_NAME"
echo "Default:          $DEFAULT_DEPLOYMENT"
echo "Deployment count: $BATCH_DEPLOYMENT_COUNT"
echo
echo "Current deployments: $BATCH_DEPLOYMENTS_JSON"
echo "Deployments to delete:   $DEPLOYMENTS_TO_DEL"

if [[ -z "$DEPLOYMENTS_TO_DEL" ]]; then
  echo "No non-default deployments found."
  exit 0
fi

DELETED_COUNT=0
DELETED_DEPLOYMENTS=()

# Read name and state for each line
while read -r DEPLOYMENT_NAME STATE; do
  echo "Checking $DEPLOYMENT_NAME (state: $STATE)"

  # Optionally skip if still provisioning
  if [[ "$STATE" != "Succeeded" && "$STATE" != "Failed" ]]; then
    echo "Skipping $DEPLOYMENT_NAME (state not deletable)."
    continue
  fi

  echo "Deleting deployment: $DEPLOYMENT_NAME"
  az ml batch-deployment delete \
    -g "$RG" -w "$WORKSPACE_NAME" -e "$BATCH_ENDPOINT_NAME" \
    -n "$DEPLOYMENT_NAME" --yes

  DELETED_DEPLOYMENTS+=("$DEPLOYMENT_NAME")
  ((DELETED_COUNT+=1))

done <<< "$DEPLOYMENTS_TO_DEL"

REMAINING_DEPLOYMENTS_JSON="$(
  az ml batch-deployment list \
    --endpoint-name "$BATCH_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query '[].name' \
    --output json
)"
REMAINING_COUNT="$(jq 'length' <<< "$REMAINING_DEPLOYMENTS_JSON")"

echo "Finished cleaning up non-default batch deployments."

{
  echo "## Batch deployment cleanup"
  echo
  echo "| Result | Count |"
  echo "|---|---:|"
  echo "| Deleted | \`$DELETED_COUNT\` |"
  echo "| Remaining | \`$REMAINING_COUNT\` |"
  echo
  echo "### Default deployment"
  echo
  echo "- \`$DEFAULT_DEPLOYMENT\`"
  echo
  echo "### Deleted deployments"
  echo
  if [[ "$DELETED_COUNT" -eq 0 ]]; then
    echo "No batch deployments were deleted."
  else
    for DEPLOYMENT_NAME in "${DELETED_DEPLOYMENTS[@]}"; do
      echo "- \`$DEPLOYMENT_NAME\`"
    done
  fi
} >> "$GITHUB_STEP_SUMMARY"