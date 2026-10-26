#!/usr/bin/env bash
# Create a new real-time deployment of a model on an existing endpoint.
#
# The deployment is named rt-deployment-<UTC timestamp>. Its name is written to
# the step output rt_deployment_name and, with the feature data version, to
# $GITHUB_ENV for later steps.
#
# Environment:
#   RT_ENDPOINT_NAME            Real-time endpoint to deploy to.
#   MODEL_NAME, MODEL_VERSION   Registered model to deploy.
#   FEATURE_DATA_ASSET_NAME     Engineered-feature data asset the scoring script reads.
#   FEATURE_DATA_ASSET_VERSION  Asset version. If empty, the latest version is used.
#   ENDPOINT_ENVIRONMENT        Azure ML environment for the deployment.
#   RT_DEPLOYMENT_YAML          Where to write the rendered deployment YAML.
#   RG, WORKSPACE_NAME          Azure ML workspace.
#   INSTANCE_TYPE               VM size (optional, default Standard_F4s_v2).
set -euo pipefail

TIMESTAMP="$(date -u +'%Y%m%d-%H%M%S')"
RT_DEPLOYMENT_NAME="rt-deployment-$TIMESTAMP"
INSTANCE_TYPE="${INSTANCE_TYPE:-Standard_F4s_v2}"
SCORE_SCRIPT="realtime.py"

echo "::group::Resolve deployment configuration"
SUBSCRIPTION_ID="$(az account show --query id --output tsv)"
if [[ -z "$SUBSCRIPTION_ID" ]]; then
  echo "::error::Could not determine the Azure subscription ID."
  exit 1
fi

if [[ -z "${FEATURE_DATA_ASSET_VERSION:-}" ]]; then
  FEATURE_DATA_ASSET_VERSION="$(
    az ml data show \
      --name "$FEATURE_DATA_ASSET_NAME" \
      --label latest \
      --resource-group "$RG" \
      --workspace-name "$WORKSPACE_NAME" \
      --query version \
      --output tsv
  )"
  if [[ -z "$FEATURE_DATA_ASSET_VERSION" ]]; then
    echo "::error title=Missing feature data version::Could not determine the engineered feature data version"
    exit 1
  fi
else
  # Check that this exact asset version exists before creating the deployment.
  az ml data show \
    --name "$FEATURE_DATA_ASSET_NAME" \
    --version "$FEATURE_DATA_ASSET_VERSION" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --output none
fi

echo "Endpoint:             $RT_ENDPOINT_NAME"
echo "Deployment:           $RT_DEPLOYMENT_NAME"
echo "Model:                ${MODEL_NAME}:${MODEL_VERSION}"
echo "Feature data:         ${FEATURE_DATA_ASSET_NAME}:${FEATURE_DATA_ASSET_VERSION}"
echo "Instance type:        $INSTANCE_TYPE"
echo "Endpoint environment: $ENDPOINT_ENVIRONMENT"
echo "Scoring script:       $SCORE_SCRIPT"
echo "Resource group:       $RG"
echo "Workspace:            $WORKSPACE_NAME"
echo "::endgroup::"

{
  echo "RT_DEPLOYMENT_NAME=$RT_DEPLOYMENT_NAME"
  echo "FEATURE_DATA_ASSET_VERSION=$FEATURE_DATA_ASSET_VERSION"
} >> "$GITHUB_ENV"
echo "rt_deployment_name=$RT_DEPLOYMENT_NAME" >> "$GITHUB_OUTPUT"

echo "::group::Render deployment YAML"
export SUBSCRIPTION_ID FEATURE_DATA_ASSET_NAME FEATURE_DATA_ASSET_VERSION \
  RT_DEPLOYMENT_NAME INSTANCE_TYPE SCORE_SCRIPT MODEL_NAME MODEL_VERSION
# shellcheck disable=SC2016  # envsubst takes the variable names literally.
envsubst \
  '${INSTANCE_TYPE}
  ${RT_DEPLOYMENT_NAME}
  ${RT_ENDPOINT_NAME}
  ${MODEL_NAME}
  ${MODEL_VERSION}
  ${ENDPOINT_ENVIRONMENT}
  ${SUBSCRIPTION_ID}
  ${RG}
  ${WORKSPACE_NAME}
  ${FEATURE_DATA_ASSET_NAME}
  ${FEATURE_DATA_ASSET_VERSION}
  ${SCORE_SCRIPT}' \
  < azureml/rt-deployment-template.yaml \
  > "$RT_DEPLOYMENT_YAML"

if [[ ! -s "$RT_DEPLOYMENT_YAML" ]]; then
  echo "::error::Deployment YAML was not generated."
  exit 1
fi

cat "$RT_DEPLOYMENT_YAML"
echo "::endgroup::"

show_logs() {
  az ml online-deployment get-logs \
    --name "$RT_DEPLOYMENT_NAME" \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --lines 200 || true
}

echo "::group::Create real-time deployment"
if ! az ml online-deployment create \
  --resource-group "$RG" \
  --workspace-name "$WORKSPACE_NAME" \
  --file "$RT_DEPLOYMENT_YAML" \
  --only-show-errors \
  --output none; then
  echo "::error title=Deployment failed::Creating '$RT_DEPLOYMENT_NAME' failed."
  show_logs
  exit 1
fi

RT_DEPLOYMENT_STATE="$(
  az ml online-deployment show \
    --name "$RT_DEPLOYMENT_NAME" \
    --endpoint-name "$RT_ENDPOINT_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query provisioning_state \
    --output tsv
)"
echo "Provisioning state: $RT_DEPLOYMENT_STATE"

if [[ "$RT_DEPLOYMENT_STATE" != "Succeeded" ]]; then
  echo "::error title=Deployment failed::'$RT_DEPLOYMENT_NAME' finished in state '$RT_DEPLOYMENT_STATE'."
  show_logs
  exit 1
fi

echo "Deployment '$RT_DEPLOYMENT_NAME' created successfully on endpoint '$RT_ENDPOINT_NAME'."
echo "::endgroup::"
