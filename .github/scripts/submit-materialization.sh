#!/usr/bin/env bash
# Submit feature-store materialization (backfill) jobs for the date window,
# and output their job IDs for the wait step.
#
# Environment:
#   FEATURE_STORE_WS                     Feature-store workspace.
#   FEATURE_SET_NAME, FEATURE_SET_VERSION  Feature set to materialize.
#   START_DATE, END_DATE, LATEST_DATE    Materialization window (DD-MM-YYYY).
#   RG                                   Resource group.
set -euo pipefail

echo ">>> Feature-store: $FEATURE_STORE_WS"
echo ">>> Feature set:   $FEATURE_SET_NAME:$FEATURE_SET_VERSION"
echo ">>> Start date:    $START_DATE"
echo ">>> Latest date:   $LATEST_DATE"
echo ">>> End date:      $END_DATE"

echo "::group::Convert materialization window"
START_DATE_ISO="$(
  python3 -c \
    'from datetime import datetime; import sys; print(datetime.strptime(sys.argv[1], "%d-%m-%Y").strftime("%Y-%m-%dT%H:%M:%SZ"))' \
    "$START_DATE"
)"

END_DATE_ISO="$(
  python3 -c \
    'from datetime import datetime; import sys; print(datetime.strptime(sys.argv[1], "%d-%m-%Y").strftime("%Y-%m-%dT%H:%M:%SZ"))' \
    "$END_DATE"
)"
echo "Materialization window:"
echo "  $START_DATE -> $START_DATE_ISO"
echo "  $END_DATE   -> $END_DATE_ISO"
echo "::endgroup::"

echo "::group::Submit materialization request"
  echo ">>> Submitting feature-store materialization."

  BACKFILL_RESP=$(az ml feature-set backfill \
    --name "$FEATURE_SET_NAME" \
    --version "$FEATURE_SET_VERSION" \
    --feature-store-name "$FEATURE_STORE_WS" \
    --resource-group "$RG" \
    --feature-window-start-time "$START_DATE_ISO" \
    --feature-window-end-time "$END_DATE_ISO" \
    --by-data-status '["None","Incomplete"]' \
    --output json)

  JOB_IDS="$(jq -c '.jobIds // []' <<< "$BACKFILL_RESP")"
  JOB_COUNT="$(jq 'length' <<< "$JOB_IDS")"

  echo "Created materialization jobs: $JOB_COUNT"
  echo "Job IDs: $JOB_IDS"

  check_active_jobs_required=false
  # The latest source date has already been materialized and there are
  # no existing active jobs to wait for.
  if [[ "$START_DATE" != "$LATEST_DATE" && "$JOB_COUNT" -eq 0 ]]; then
    echo "Active jobs need to be checked as:"
    echo "- Start date $START_DATE != latest date $LATEST_DATE"
    echo "- Job count $JOB_COUNT equals 0."
    check_active_jobs_required=true
  fi
  echo "::endgroup::"

  echo "::group::Check active materialization jobs"

  if [[ "$check_active_jobs_required" == true ]]; then

    echo "No new jobs were returned by the backfill request."
    echo "Checking for existing active materialization jobs as:."

    ACTIVE_JOB_IDS="$(
      az ml job list \
        --resource-group "$RG" \
        --workspace-name "$FEATURE_STORE_WS" \
        --query "[?starts_with(name, 'fs-mat-') && contains(['NotStarted','Starting','Preparing','Queued','Running','Finalizing'], status)].name" \
        --output tsv
    )"

    MATCHING_JOB_IDS='[]'

    if [[ -z "$ACTIVE_JOB_IDS" ]]; then
      echo "No active materialization jobs found."
    else
      echo "Active materialization jobs:"
      echo "$ACTIVE_JOB_IDS"

      while IFS= read -r JOB_ID; do
        [[ -z "$JOB_ID" ]] && continue

        echo
        echo "Checking job: $JOB_ID"

        JOB_ARGS="$(
          az ml job show \
            --name "$JOB_ID" \
            --resource-group "$RG" \
            --workspace-name "$FEATURE_STORE_WS" \
            --query args \
            --output tsv
        )"

        JOB_FEATURE_STORE="$(
          sed -n 's/.*--featurestore-name \([^ ]*\).*/\1/p' <<< "$JOB_ARGS"
        )"

        JOB_FEATURE_SET="$(
          sed -n 's/.*--featureset-name \([^ ]*\).*/\1/p' <<< "$JOB_ARGS"
        )"

        JOB_FEATURE_SET_VERSION="$(
          sed -n 's/.*--featureset-version \([^ ]*\).*/\1/p' <<< "$JOB_ARGS"
        )"

        echo "Job:"
        echo "  Feature store: $JOB_FEATURE_STORE"
        echo "  Feature set:   $JOB_FEATURE_SET"
        echo "  Version:       $JOB_FEATURE_SET_VERSION"

        if [[ "$JOB_FEATURE_STORE" == "$FEATURE_STORE_WS" &&
              "$JOB_FEATURE_SET" == "$FEATURE_SET_NAME" &&
              "$JOB_FEATURE_SET_VERSION" == "$FEATURE_SET_VERSION" ]]; then

          echo ">>> Matching active materialization job found: $JOB_ID"

          MATCHING_JOB_IDS="$(
            jq -c \
              --arg job_id "$JOB_ID" \
              '. + [$job_id]' \
              <<< "$MATCHING_JOB_IDS"
          )"
        else
          echo "Job belongs to a different feature set."
        fi

      done <<< "$ACTIVE_JOB_IDS"
    fi

    MATCHING_JOB_COUNT="$(jq 'length' <<< "$MATCHING_JOB_IDS")"

    echo
    echo "Matching active jobs: $MATCHING_JOB_COUNT"
    echo "Job IDs: $MATCHING_JOB_IDS"

    if [[ "$MATCHING_JOB_COUNT" -eq 0 ]]; then
      echo "::error title=Missing materialization job::Backfill returned no job IDs and no matching active materialization jobs were found."
      exit 1
    fi

    JOB_IDS="$MATCHING_JOB_IDS"
    JOB_COUNT="$MATCHING_JOB_COUNT"

  else
    echo "Active-job check not required as either:"
    echo "- Job count $JOB_COUNT does not equal 0"
    echo "- Start date $START_DATE and latest date $LATEST_DATE are equal"
  fi

  echo "::endgroup::"

echo "::group::Materialization result"

echo "Materialization jobs to monitor: $JOB_COUNT"
echo "Job IDs: $JOB_IDS"

{
  echo "job_ids=$JOB_IDS"
  echo "job_count=$JOB_COUNT"
} >> "$GITHUB_OUTPUT"

echo "::endgroup::"
