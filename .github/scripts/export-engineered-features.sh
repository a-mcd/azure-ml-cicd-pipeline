#!/usr/bin/env bash
# Produce the engineered-features CSV to register as a data asset.
# With the feature store, export every materialized week from the offline
# store; with BACKFILL_WORKING=false, validate the CSV built by the
# engineered-features job instead.
#
# Environment:
#   BACKFILL_WORKING                      "false" when the feature store is not used.
#   ENGINEERED_DATA_PATH                  CSV to write or validate.
#   ENGINEERED_DATA_VERSION               Version the asset will be registered as.
#   FEATURE_STORE_WS                      Feature-store workspace.
#   FEATURE_SET_NAME, FEATURE_SET_VERSION  Feature set to export.
#   RG                                    Resource group.
set -euo pipefail

if [[ "${BACKFILL_WORKING:-}" == "false" ]]; then
  echo "No feature store: validating $ENGINEERED_DATA_PATH built by the engineered-features job."
else

echo "::group::Export configuration"

echo "Resource group:        $RG"
echo "Feature store:         $FEATURE_STORE_WS"
echo "Feature set:           $FEATURE_SET_NAME:$FEATURE_SET_VERSION"
echo "Export destination:    $ENGINEERED_DATA_PATH"
echo "Asset version:         $ENGINEERED_DATA_VERSION"

echo "::endgroup::"

echo "::group::Discover feature-store offline storage"
rm -f "$ENGINEERED_DATA_PATH"

SUBSCRIPTION_ID="$(
    az account show \
        --query id \
        --output tsv
)"

FEATURE_STORE_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG}/providers/Microsoft.MachineLearningServices/workspaces/${FEATURE_STORE_WS}"

FEATURE_STORE_STORAGE_ID="$(
    az resource show \
        --ids "$FEATURE_STORE_ID" \
        --query properties.storageAccount \
        --output tsv
)"

OFFLINE_STORE_SCOPE="$(
    az ml feature-store show \
        --name "$FEATURE_STORE_WS" \
        --resource-group "$RG" \
        --query offline_store.target \
        --output tsv
)"

if [[ -z "$FEATURE_STORE_STORAGE_ID" || \
      "$FEATURE_STORE_STORAGE_ID" == "null" ]]; then
    echo "ERROR: Could not determine the feature-store storage resource ID." >&2
    exit 1
fi

if [[ -z "$OFFLINE_STORE_SCOPE" ||
      "$OFFLINE_STORE_SCOPE" == "null" ]]; then
    echo "ERROR: Could not determine the offline-store container." >&2
    exit 1
fi

# Extract the names from their Azure resource IDs.
export OFFLINE_STORAGE_ACCOUNT="${FEATURE_STORE_STORAGE_ID##*/}"
export OFFLINE_STORAGE_CONTAINER="${OFFLINE_STORE_SCOPE##*/}"

echo "Offline storage account: $OFFLINE_STORAGE_ACCOUNT"
echo "Offline storage container: $OFFLINE_STORAGE_CONTAINER"
echo "::endgroup::"

echo "::group::Retrieve offline-storage credentials"
export AZURE_STORAGE_KEY="$(
    az storage account keys list \
        --resource-group "$RG" \
        --account-name "$OFFLINE_STORAGE_ACCOUNT" \
        --query "[0].value" \
        --output tsv
)"
echo "::add-mask::$AZURE_STORAGE_KEY"

if [[ -z "${AZURE_STORAGE_KEY:-}" || \
      "$AZURE_STORAGE_KEY" == "null" ]]; then
    echo "ERROR: Could not retrieve the feature-store storage key." >&2
    exit 1
fi
echo "::endgroup::"

echo "::group::Export materialized features"
echo "Exporting all materialized feature data..."
python3 src/walmart_ml/pipeline/create_all_weeks_features.py
echo "::endgroup::"
fi

echo "::group::Validate materialized-feature export"
if [[ ! -f "$ENGINEERED_DATA_PATH" ]]; then
  echo "::error title=Missing feature export::$ENGINEERED_DATA_PATH was not created"
  exit 1
fi

if [[ ! -s "$ENGINEERED_DATA_PATH" ]]; then
  echo "::error title=Empty feature export::$ENGINEERED_DATA_PATH is empty"
  exit 1
fi

EXPORT_DETAILS="$(
  python3 - "$ENGINEERED_DATA_PATH" <<'PYTHON'
import json
import sys

import pandas as pd

output_path = sys.argv[1]
dataframe = pd.read_csv(output_path)

if dataframe.empty:
    raise RuntimeError(
        "The materialized-feature export contains no rows."
    )

required_columns = {"Store", "Date"}
missing_columns = sorted(
    required_columns.difference(dataframe.columns)
)

if missing_columns:
    raise RuntimeError(
        "The materialized-feature export is missing columns: "
        + ", ".join(missing_columns)
    )

dataframe["Date"] = pd.to_datetime(
    dataframe["Date"],
    format="%d-%m-%Y",
    errors="raise",
)

details = {
    "rows": int(len(dataframe)),
    "stores": int(dataframe["Store"].nunique()),
    "minimum_date": dataframe["Date"].min().strftime("%d-%m-%Y"),
    "maximum_date": dataframe["Date"].max().strftime("%d-%m-%Y"),
}

print(json.dumps(details))
PYTHON
)"

echo "$EXPORT_DETAILS" | jq .

{
  echo "row_count=$(jq -r '.rows' <<< "$EXPORT_DETAILS")"
  echo "store_count=$(jq -r '.stores' <<< "$EXPORT_DETAILS")"
  echo "minimum_date=$(jq -r '.minimum_date' <<< "$EXPORT_DETAILS")"
  echo "maximum_date=$(jq -r '.maximum_date' <<< "$EXPORT_DETAILS")"
} >> "$GITHUB_OUTPUT"
echo "::endgroup::"
