#!/usr/bin/env bash
# Register a CSV file as a uri_file data asset at a fixed version, then read
# it back to check the version, and write a job summary.
#
# Environment:
#   ASSET_NAME, ASSET_VERSION  Data asset to create.
#   DATA_FILE                  CSV file to upload.
#   RG, WORKSPACE_NAME         Workspace to register the asset in.
#   LABEL                      Description for messages, for example "test data".
set -euo pipefail

echo "::group::Validate $LABEL artifact"
echo "File: $DATA_FILE"

if [[ ! -s "$DATA_FILE" ]]; then
  echo "::error title=Missing $LABEL::File not found or empty: $DATA_FILE"
  echo
  echo "Available files:"
  find . -maxdepth 3 -type f -print
  exit 1
fi

FILE_SIZE="$(du -h "$DATA_FILE" | cut -f1)"
TOTAL_LINES="$(wc -l < "$DATA_FILE")"

if (( TOTAL_LINES <= 1 )); then
  echo "::error title=Empty $LABEL::The CSV contains no data rows"
  exit 1
fi

ROW_COUNT=$(( TOTAL_LINES - 1 ))
echo "Size:      $FILE_SIZE"
echo "Data rows: $ROW_COUNT"
echo "::endgroup::"

echo "::group::Register $LABEL"
CREATED_VERSION="$(
  az ml data create \
    --name "$ASSET_NAME" \
    --version "$ASSET_VERSION" \
    --path "$DATA_FILE" \
    --type uri_file \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --only-show-errors \
    --output json |
  jq -r '.version // empty'
)"

echo "Expected version: $ASSET_VERSION"
echo "Created version:  $CREATED_VERSION"

if [[ -z "$CREATED_VERSION" ]]; then
  echo "::error title=Registration failed::Azure ML did not return a data asset version"
  exit 1
fi

if [[ "$CREATED_VERSION" != "$ASSET_VERSION" ]]; then
  echo "::error title=Data version mismatch::Azure ML returned version '$CREATED_VERSION', expected '$ASSET_VERSION'"
  exit 1
fi
echo "::endgroup::"

echo "::group::Verify $LABEL"
REGISTERED_DATA_JSON="$(
  az ml data show \
    --name "$ASSET_NAME" \
    --version "$ASSET_VERSION" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --output json \
    --only-show-errors
)"
REGISTERED_VERSION="$(jq -r '.version // empty' <<< "$REGISTERED_DATA_JSON")"
REGISTERED_PATH="$(jq -r '.path // empty' <<< "$REGISTERED_DATA_JSON")"

if [[ "$REGISTERED_VERSION" != "$ASSET_VERSION" ]]; then
  echo "::error title=Verification failed::Retrieved version '${REGISTERED_VERSION:-none}' of $ASSET_NAME, expected '$ASSET_VERSION'"
  exit 1
fi

echo "Data asset: $ASSET_NAME"
echo "Version:    $REGISTERED_VERSION"
echo "Path:       $REGISTERED_PATH"
echo "::endgroup::"

{
  echo "## Upload of $LABEL"
  echo
  echo "| Property | Value |"
  echo "|---|---|"
  echo "| Resource group | \`$RG\` |"
  echo "| Workspace | \`$WORKSPACE_NAME\` |"
  echo "| Data asset | \`$ASSET_NAME\` |"
  echo "| Version | \`$REGISTERED_VERSION\` |"
  echo "| Type | \`uri_file\` |"
  echo "| Data rows | \`$ROW_COUNT\` |"
  echo "| File size | \`$FILE_SIZE\` |"
  echo "| Registered path | \`$REGISTERED_PATH\` |"
} >> "$GITHUB_STEP_SUMMARY"
