#!/usr/bin/env bash
# Work out the next version number shared by the Data Lake file, the
# engineered-data asset and the test-data asset: one more than the highest
# existing version of any of them.
#
# Environment:
#   ADLS_ACCOUNT, ADLS_FILESYSTEM, ADLS_VERSION_ROOT, ADLS_FILENAME
#                               Data Lake location of the versioned source CSVs.
#   ENGINEERED_DATA_ASSET_NAME  Engineered-data asset name.
#   RG, WORKSPACE_NAME          Azure ML workspace.
set -euo pipefail

echo "::group::Check ADLS versions"
# Find the latest numeric ADLS directory.
VERSION_DIRECTORIES="$(
  az storage fs directory list \
    --account-name "$ADLS_ACCOUNT" \
    --file-system "$ADLS_FILESYSTEM" \
    --path "$ADLS_VERSION_ROOT" \
    --auth-mode login \
    --query "[?isDirectory == \`true\`].name" \
    --output tsv
)"

LATEST_ADLS_VERSION="$(
  awk -F/ '
    $NF ~ /^[0-9]+$/ {
      version = $NF + 0

      if (version > maximum) {
        maximum = version
      }
    }
    END {
      print maximum + 0
    }
  ' <<< "$VERSION_DIRECTORIES"
)"
echo "::endgroup::"

echo "::group::Check engineered-data asset versions"
# Find the latest numeric engineered-data asset version.
ENGINEERED_VERSIONS="$(
  az ml data list \
    --name "$ENGINEERED_DATA_ASSET_NAME" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "[].version" \
    --output tsv
)"

LATEST_ENGINEERED_VERSION="$(
  awk '
    /^[0-9]+$/ {
      version = $1 + 0

      if (version > maximum) {
        maximum = version
      }
    }
    END {
      print maximum + 0
    }
  ' <<< "$ENGINEERED_VERSIONS"
)"
echo "::endgroup::"

echo "::group::Check test-data asset versions"
# Find the latest numeric test-data asset version.
TEST_DATA_VERSIONS="$(
  az ml data list \
    --name "test_data" \
    --resource-group "$RG" \
    --workspace-name "$WORKSPACE_NAME" \
    --query "[].version" \
    --output tsv
)"

LATEST_TEST_DATA_VERSION="$(
  awk '
    /^[0-9]+$/ {
      version = $1 + 0

      if (version > maximum) {
        maximum = version
      }
    }
    END {
      print maximum + 0
    }
  ' <<< "$TEST_DATA_VERSIONS"
)"
echo "::endgroup::"

echo "::group::Calculate next shared version"
# Choose a version that does not already exist in either place.
LATEST_SHARED_VERSION="$(
  printf '%s\n' \
    "$LATEST_ADLS_VERSION" \
    "$LATEST_ENGINEERED_VERSION" \
    "$LATEST_TEST_DATA_VERSION" |
    sort -n |
    tail -n 1
)"

SHARED_VERSION=$((LATEST_SHARED_VERSION + 1))
ADLS_FILE_VERSION_PATH="${ADLS_VERSION_ROOT}/${SHARED_VERSION}/${ADLS_FILENAME}"

{
  echo "shared_version=$SHARED_VERSION"
  echo "latest_adls_version=$LATEST_ADLS_VERSION"
  echo "latest_engineered_version=$LATEST_ENGINEERED_VERSION"
  echo "latest_test_data_version=$LATEST_TEST_DATA_VERSION"
  echo "adls_file_version_path=$ADLS_FILE_VERSION_PATH"
} >> "$GITHUB_OUTPUT"

echo ">>> Latest ADLS version:            $LATEST_ADLS_VERSION"
echo ">>> Latest engineered-data version: $LATEST_ENGINEERED_VERSION"
echo ">>> Latest test-data version:       $LATEST_TEST_DATA_VERSION"
echo ">>> Next shared version:            $SHARED_VERSION"
echo ">>> New ADLS path:                  $ADLS_FILE_VERSION_PATH"
echo "::endgroup::"

echo "::group::Set job outputs"

{
  echo "shared_version=$SHARED_VERSION"
  echo "latest_adls_version=$LATEST_ADLS_VERSION"
  echo "latest_engineered_version=$LATEST_ENGINEERED_VERSION"
  echo "latest_test_data_version=$LATEST_TEST_DATA_VERSION"
  echo "adls_file_version_path=$ADLS_FILE_VERSION_PATH"
} >> "$GITHUB_OUTPUT"

echo "Outputs successfully set."
echo "::endgroup::"

{
  echo "## Shared data version"
  echo
  echo "| Location | Latest version |"
  echo "|---|---:|"
  echo "| ADLS | \`$LATEST_ADLS_VERSION\` |"
  echo "| Engineered data | \`$LATEST_ENGINEERED_VERSION\` |"
  echo "| Test data | \`$LATEST_TEST_DATA_VERSION\` |"
  echo
  echo "### New version"
  echo
  echo "- **Shared version:** \`$SHARED_VERSION\`"
  echo "- **ADLS path:** \`$ADLS_FILE_VERSION_PATH\`"
} >> "$GITHUB_STEP_SUMMARY"
