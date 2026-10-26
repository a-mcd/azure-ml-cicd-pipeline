"""Shared constants for training, scoring and the pipeline scripts.

Keeping these values in one place means the training code and the scripts
that read its outputs always agree on column names and split sizes.
"""

METADATA_COLUMNS = ["Store", "Date", "Weekly_Sales", "Holiday"]
TEST_HORIZON_WEEKS = 52
CHALLENGER_HORIZON_WEEKS = 1
ROLL_WINDOWS = (3, 6, 12)
RANDOM_STATE = 42
DEFAULT_REGISTRY_URI = "file:///tmp/mlflow-registry"

# Columns in challenger_forecast_by_store.csv. Written by training/outputs.py
# and read back by pipeline/select_winner.py, so both must use this list.
CHALLENGER_FORECAST_COLUMNS = [
    "Store",
    "Date",
    "Next_Week_Date",
    "Weekly_Sales",
    "Actual_Next_Week_Sales",
    "Predicted_Weekly_Sales",
]
