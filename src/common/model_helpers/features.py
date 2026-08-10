import pandas as pd
from src.common.model_helpers import config

def prepare_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features from the Date column.

    These features describe the current row's week number/month/quarter and help the model learn
    seasonal sales patterns.

    Might be beneficial to have target date calendar features instead of current week.

    Returns:
        pd.DataFrame: Dataset with additional calendar feature columns:
        year, month, week, and quarter.

    """
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter

    return df


def prepare_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add holiday features based on the Holiday_Flag and month.

    The Holiday feature is labeled as the current-week holiday, not the target-week holiday.
    This may be beneficial for the model to learn from current-week holiday effects.

    Returns:
        pd.DataFrame: Dataset with an additional Holiday feature column.
    """
    df["Holiday"] = "N/A"
    m = df.get("Holiday_Flag", 0) == 1  # handle if column missing

    df.loc[m & (df["month"] == 11), "Holiday"] = "Thanksgiving"
    df.loc[m & (df["month"] == 12), "Holiday"] = "Christmas"
    df.loc[m & (df["month"] == 2), "Holiday"] = "Super_Bowl"
    df.loc[m & (df["month"] == 9), "Holiday"] = "Labor_Day"

    # Drop redundant binary flag
    df = df.drop(columns=["Holiday_Flag"], errors='ignore')

    return df


def prepare_time_series_features(df: pd.DataFrame, drop_training_rows: bool = True) -> pd.DataFrame:
    """
    Add per-store time-series features and the next-week sales target.

    The dataset is sorted by Store and Date before creating features. Lag
    features and rolling means are calculated separately for each store to
    avoid mixing sales history between stores. Rolling means use shifted sales
    values so the current week's sales are not included, which helps prevent
    data leakage.

    Returns:
        pd.DataFrame: Dataset with lag features, rolling mean features,
        the next-week sales target, and unusable training rows removed.
    """

    df = df.sort_values(["Store", "Date"])

    # Add lag features: last week, two weeks ago, and same week last year
    for lag in (1, 2, 52):
        df[f"lag_{lag}"] = df.groupby("Store")["Weekly_Sales"].shift(lag)

    # Add rolling mean features using only previous weeks' sales
    for w in config.ROLL_WINDOWS:
        df[f"rollmean_{w}"] = (
            df.groupby("Store")["Weekly_Sales"]
            .apply(
                lambda s, window=w: s.shift(1).rolling(window=window, min_periods=1).mean()
            )
            .reset_index(level=0, drop=True)
        )

    # Target: next week's sales for same store (shift -1)
    df["Weekly_Sales_tplus1"] = df.groupby("Store")["Weekly_Sales"].shift(-1)

    # Remove rows without a target or previous-weeks sales value
    if drop_training_rows:
        df = df.dropna(subset=["Weekly_Sales_tplus1", "lag_1", "lag_2", "lag_52"]).reset_index(drop=True)

    return df


def build_features(df: pd.DataFrame, drop_training_rows: bool = True) -> pd.DataFrame:
    """
    Build the full feature set for next-week Walmart sales prediction.

    Creates a copy of the input dataset, ensures Store is stored as an integer,
    then adds calendar, holiday, and time-series features. The returned dataset
    is ready for later preprocessing, model training, and evaluation.

    Returns:
        pd.DataFrame: Dataset with engineered calendar, holiday, lag, rolling
        mean, and next-week target features.
    """
    df = df.copy()
    df["Store"] = df["Store"].astype(int) # ensure Store is integer type for grouping and encoding

    df = prepare_calendar_features(df)
    df = prepare_holiday_features(df)
    df = prepare_time_series_features(df, drop_training_rows)

    return df


def align_to_feature_columns(
    df_enc: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Align an encoded dataframe to the exact feature columns expected by the model.

    Missing columns are added with 0.
    Extra columns are dropped.
    Column order is matched to feature_cols.
    """
    return df_enc.reindex(columns=feature_cols, fill_value=0)
