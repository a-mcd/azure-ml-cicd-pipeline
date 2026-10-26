#!/usr/bin/env bash
# Wait for an Azure ML job to finish.
#
# Logs are streamed for visibility only. 'az ml job stream' can crash while
# decoding logs (e.g. "Invalid base64-encoded string") even when the job
# succeeds, so the job's own status decides whether this script passes.
#
# Environment:
#   JOB_NAME         Azure ML job name.
#   RG               Resource group of the workspace.
#   WORKSPACE_NAME   Azure ML workspace.
#   TIMEOUT_MINUTES  Minutes to wait before cancelling the job.
#   JOB_DESCRIPTION  What the job does, for messages (optional).
set -euo pipefail

JOB_DESCRIPTION="${JOB_DESCRIPTION:-Azure ML job}"
TIMEOUT_SECONDS=$(( TIMEOUT_MINUTES * 60 ))
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))

echo "::group::Stream logs for $JOB_NAME"
timeout "$TIMEOUT_SECONDS" \
  az ml job stream --name "$JOB_NAME" --resource-group "$RG" --workspace-name "$WORKSPACE_NAME" \
  || echo "::warning title=Log streaming stopped::Checking the status of $JOB_NAME instead."
echo "::endgroup::"

while true; do
  JOB_STATUS="$(
    az ml job show \
      --name "$JOB_NAME" \
      --resource-group "$RG" \
      --workspace-name "$WORKSPACE_NAME" \
      --query status \
      --output tsv
  )"
  case "$JOB_STATUS" in
    Completed)
      break
      ;;
    Failed|Canceled|Cancelled|NotResponding)
      echo "::error title=$JOB_DESCRIPTION failed::$JOB_NAME finished with status '$JOB_STATUS'"
      exit 1
      ;;
    *)
      if (( $(date +%s) >= DEADLINE )); then
        echo "::error title=Azure ML job timed out::$JOB_NAME still '$JOB_STATUS' after $TIMEOUT_MINUTES minutes. Cancelling it."
        az ml job cancel --name "$JOB_NAME" --resource-group "$RG" --workspace-name "$WORKSPACE_NAME" \
          || echo "::warning title=Cancel failed::Could not cancel $JOB_NAME; cancel it in Azure ML Studio."
        exit 1
      fi
      echo "Job status: $JOB_STATUS. Waiting..."
      sleep 30
      ;;
  esac
done

echo "$JOB_DESCRIPTION $JOB_NAME completed successfully."
