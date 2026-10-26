#!/usr/bin/env bash
# Download a batch job's predictions and validate them.
#
# Needs the repository checked out and walmart_ml's dependencies installed,
# because it runs src/walmart_ml/pipeline/validate_predictions.py.
#
# Environment:
#   BATCH_JOB_NAME  Batch scoring job name.
#   RG              Resource group of the workspace.
#   WORKSPACE_NAME  Azure ML workspace.
#   DOWNLOAD_PATH   Folder to download the predictions into.
set -euo pipefail

PREDICTIONS_FILE="$DOWNLOAD_PATH/predictions.csv"

echo "::group::Download batch predictions"
az ml job download \
  --name "$BATCH_JOB_NAME" \
  --output-name score \
  --download-path "$DOWNLOAD_PATH" \
  --resource-group "$RG" \
  --workspace-name "$WORKSPACE_NAME"

if [[ ! -f "$PREDICTIONS_FILE" ]]; then
  echo "::error title=Missing predictions::File not found: $PREDICTIONS_FILE"
  exit 1
fi

if [[ ! -s "$PREDICTIONS_FILE" ]]; then
  echo "::error title=Empty predictions::Predictions file is empty."
  exit 1
fi

echo "File found: $PREDICTIONS_FILE"
# The append_row output has no header row, so every non-empty line is a prediction.
PREDICTION_COUNT="$(grep -c . "$PREDICTIONS_FILE")"
{
  echo "prediction_count=$PREDICTION_COUNT"
  echo "predictions-file=$PREDICTIONS_FILE"
} >> "$GITHUB_OUTPUT"

cat "$PREDICTIONS_FILE"
echo "::endgroup::"

echo "::group::Validate predictions"
if ! python src/walmart_ml/pipeline/validate_predictions.py "$PREDICTIONS_FILE"; then
  echo "::error title=Prediction validation failed::$PREDICTIONS_FILE failed validation"
  exit 1
fi
echo "::endgroup::"
