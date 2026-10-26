#!/usr/bin/env bash
# Make a deployment the batch endpoint's default.
#
# Environment:
#   BATCH_ENDPOINT_NAME    Batch endpoint name.
#   BATCH_DEPLOYMENT_NAME  Deployment to make the default.
#   RG                     Resource group of the workspace.
#   WORKSPACE_NAME         Azure ML workspace.
set -euo pipefail
echo "::group::Current default batch deployment"
PREVIOUS_DEFAULT="$(
  az ml batch-endpoint show \
    --name "$BATCH_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query defaults.deployment_name \
    --output tsv
)"

echo "Previous default: ${PREVIOUS_DEFAULT:-none}"
echo "New default:      $BATCH_DEPLOYMENT_NAME"
echo "::endgroup::"

echo "::group::Update default batch deployment"
az ml batch-endpoint update \
  --name "$BATCH_ENDPOINT_NAME" \
  --resource-group "$RG" \
  --workspace-name "$WORKSPACE_NAME" \
  --set "defaults.deployment_name=$BATCH_DEPLOYMENT_NAME" \
  --output none
echo "::endgroup::"

CURRENT_DEFAULT="$(
  az ml batch-endpoint show \
    --name "$BATCH_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query defaults.deployment_name \
    --output tsv
)"

if [[ "$CURRENT_DEFAULT" != "$BATCH_DEPLOYMENT_NAME" ]]; then
  echo "::error title=Default deployment update failed::Expected '$BATCH_DEPLOYMENT_NAME', found '${CURRENT_DEFAULT:-none}'"
  exit 1
fi

{
  echo "previous-default=${PREVIOUS_DEFAULT:-none}"
  echo "current-default=$CURRENT_DEFAULT"
} >> "$GITHUB_OUTPUT"
