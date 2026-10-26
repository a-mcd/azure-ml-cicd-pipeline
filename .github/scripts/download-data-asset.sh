#!/usr/bin/env bash
# Download the file behind a uri_file data asset from its workspace's blob
# datastore, check it has data rows, and write a job summary.
#
# Environment:
#   ASSET_NAME, ASSET_VERSION  Data asset to download.
#   SOURCE_RG                  Resource group of the source workspace.
#   SOURCE_WORKSPACE_NAME      Workspace that holds the asset.
#   LOCAL_FILE                 Where to save the file.
#   LABEL                      Description for messages, for example "feature data".
#   ENV_PREFIX                 Optional. Also writes <ENV_PREFIX>_FILE_SIZE,
#                              <ENV_PREFIX>_ROW_COUNT and <ENV_PREFIX>_SOURCE_PATH
#                              to $GITHUB_ENV.
set -euo pipefail

DATASTORE_NAME="workspaceblobstore"

echo "::group::Resolve ${ASSET_NAME}:${ASSET_VERSION}"
ASSET_PATH="$(
  az ml data show \
    --name "$ASSET_NAME" \
    --version "$ASSET_VERSION" \
    --resource-group "$SOURCE_RG" \
    --workspace-name "$SOURCE_WORKSPACE_NAME" \
    --query path \
    --output tsv
)"

if [[ -z "$ASSET_PATH" || "$ASSET_PATH" == "null" ]]; then
  echo "::error title=Missing $LABEL path::Could not resolve the path for ${ASSET_NAME}:${ASSET_VERSION}"
  exit 1
fi

BLOB_NAME="${ASSET_PATH#*/paths/}"

datastore_field() {
  az ml datastore show \
    --name "$DATASTORE_NAME" \
    --resource-group "$SOURCE_RG" \
    --workspace-name "$SOURCE_WORKSPACE_NAME" \
    --query "$1" \
    --output tsv
}

STORAGE_ACCOUNT_NAME="$(datastore_field account_name)"
CONTAINER_NAME="$(datastore_field container_name)"

if [[ -z "$STORAGE_ACCOUNT_NAME" || "$STORAGE_ACCOUNT_NAME" == "null" ]]; then
  echo "::error title=Missing storage account::Could not resolve the datastore storage account."
  exit 1
fi

if [[ -z "$CONTAINER_NAME" || "$CONTAINER_NAME" == "null" ]]; then
  echo "::error title=Missing container::Could not resolve the datastore container."
  exit 1
fi

echo "Datastore:       $DATASTORE_NAME"
echo "Storage account: $STORAGE_ACCOUNT_NAME"
echo "Container:       $CONTAINER_NAME"
echo "Blob:            $BLOB_NAME"
echo "::endgroup::"

echo "::group::Download $LABEL"
mkdir -p "$(dirname "$LOCAL_FILE")"

az storage blob download \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --container-name "$CONTAINER_NAME" \
  --name "$BLOB_NAME" \
  --file "$LOCAL_FILE" \
  --overwrite true \
  --only-show-errors
echo "::endgroup::"

echo "::group::Validate downloaded $LABEL"
if [[ ! -s "$LOCAL_FILE" ]]; then
  echo "::error title=Download failed::The downloaded $LABEL file is missing or empty."
  exit 1
fi

FILE_SIZE="$(du -h "$LOCAL_FILE" | cut -f1)"
TOTAL_LINES="$(wc -l < "$LOCAL_FILE")"
# The CSV has a header row.
ROW_COUNT=$(( TOTAL_LINES > 0 ? TOTAL_LINES - 1 : 0 ))

echo "File:      $LOCAL_FILE"
echo "Size:      $FILE_SIZE"
echo "Data rows: $ROW_COUNT"

if (( ROW_COUNT == 0 )); then
  echo "::error title=Empty $LABEL::The downloaded CSV contains no data rows"
  exit 1
fi
echo "::endgroup::"

if [[ -n "${ENV_PREFIX:-}" ]]; then
  {
    echo "${ENV_PREFIX}_FILE_SIZE=$FILE_SIZE"
    echo "${ENV_PREFIX}_ROW_COUNT=$ROW_COUNT"
    echo "${ENV_PREFIX}_SOURCE_PATH=$ASSET_PATH"
  } >> "$GITHUB_ENV"
fi

{
  echo "## Download of $LABEL"
  echo
  echo "| Property | Value |"
  echo "|---|---|"
  echo "| Source workspace | \`$SOURCE_WORKSPACE_NAME\` |"
  echo "| Data asset | \`$ASSET_NAME\` |"
  echo "| Version | \`$ASSET_VERSION\` |"
  echo "| Data rows | $ROW_COUNT |"
  echo "| Size | $FILE_SIZE |"
} >> "$GITHUB_STEP_SUMMARY"
