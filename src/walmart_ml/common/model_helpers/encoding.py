import pandas as pd


def prepare_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns for model training, testing, or forecasting.

    Converts Store into binary store columns using the prefix 'S_' and converts
    Holiday into binary holiday columns using the prefix 'Holiday_'.

    Missing holiday values are treated as 'N/A', and known holiday categories are
    fixed to keep holiday encoding consistent across training, testing, and
    forecasting.

    Args:
        df (pd.DataFrame): Dataset before categorical encoding.

    Returns:
        pd.DataFrame: Encoded dataset.
    """
    df = df.copy()

    known_holidays = ["N/A", "Christmas", "Thanksgiving", "Labor_Day", "Super_Bowl"]

    df["Holiday"] = df["Holiday"].fillna("N/A")
    df["Holiday"] = pd.Categorical(df["Holiday"], categories=known_holidays)

    df_enc = pd.get_dummies(
        df,
        columns=["Store", "Holiday"],
        prefix={"Store": "S", "Holiday": "Holiday"},
        drop_first=False,
    )

    df_enc = df_enc.drop(columns=["Holiday_N/A"], errors="ignore")

    return df_enc


def get_feature_columns(train_enc: pd.DataFrame) -> list[str]:
    """
    Build the final list of feature columns used as model inputs.

    Includes manually selected numeric features and any one-hot encoded Store
    or Holiday columns created during preprocessing.

    Args:
        train_enc (pd.DataFrame): Encoded training dataset.

    Returns:
        list[str]: Column names to use as model input features.
    """
    base_features = [
        "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "lag_1", "lag_2", "lag_52",
        "rollmean_3", "rollmean_6", "rollmean_12",
        "month", "week", "quarter", "year",
    ]

    onehot_cols = [
        col for col in train_enc.columns
        if col.startswith("S_") or col.startswith("Holiday_")
    ]

    return base_features + onehot_cols
