#!/usr/bin/env bash
# Wait until a Data Lake file exists with the same size as the local file it
# was uploaded from. Retries every 30 seconds, up to MAX_ATTEMPTS times.
#
# Environment:
#   ADLS_ACCOUNT, ADLS_FILESYSTEM  Data Lake account and filesystem.
#   REMOTE_PATH                    Path of the file in the Data Lake.
#   LOCAL_FILE                     File that was uploaded.
#   LABEL                          Description for messages, for example "current".
#   MAX_ATTEMPTS                   Optional, default 20.
set -euo pipefail

MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
LOCAL_FILE_SIZE="$(stat --format='%s' "$LOCAL_FILE")"

echo "Path:             $REMOTE_PATH"
echo "Expected size:    $LOCAL_FILE_SIZE bytes"
echo "Maximum attempts: $MAX_ATTEMPTS"
echo "Retry interval:   30 seconds"

for (( attempt = 1; attempt <= MAX_ATTEMPTS; attempt++ )); do
  if FILE_DETAILS="$(
    az storage fs file show \
      --account-name "$ADLS_ACCOUNT" \
      --file-system "$ADLS_FILESYSTEM" \
      --path "$REMOTE_PATH" \
      --auth-mode login \
      --output json 2>/dev/null
  )"; then
    REMOTE_FILE_SIZE="$(jq -r '.size' <<< "$FILE_DETAILS")"

    echo ">>> Local file size:  $LOCAL_FILE_SIZE bytes"
    echo ">>> Remote file size: $REMOTE_FILE_SIZE bytes"

    if [[ "$REMOTE_FILE_SIZE" == "$LOCAL_FILE_SIZE" ]]; then
      echo ">>> The $LABEL Data Lake file has the expected size."
      jq '{path: .name, size: .contentLength, lastModified: .lastModified}' <<< "$FILE_DETAILS"
      exit 0
    fi

    echo ">>> The $LABEL Data Lake file does not yet have the expected size."
  fi

  echo ">>> Waiting for the $LABEL Data Lake file: attempt $attempt/$MAX_ATTEMPTS"
  if (( attempt < MAX_ATTEMPTS )); then
    sleep 30
  fi
done

echo "::error title=Data Lake verification failed::$REMOTE_PATH is missing, inaccessible, or has the wrong size"
exit 1
