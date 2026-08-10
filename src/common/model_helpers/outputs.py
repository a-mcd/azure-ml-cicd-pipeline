import matplotlib.pyplot as plt
import pandas as pd

def output_train_test_sets(*, X_train, y_train, X_test, y_test, train_meta, test_meta, outputs_dir) -> None:
    """
    Save the prepared training and test datasets to CSV files.

    Combines metadata, model features and target values, then writes the resulting
    training and test DataFrames to the specified outputs directory. This is
    useful for debugging, auditing, and keeping a record of the exact data used
    for model training and evaluation.

    Args:
        X_train: Training feature dataset.
        y_train: Training target values.
        X_test: Test feature dataset.
        y_test: Test target values.
        train_meta: Metadata for the training dataset.
        test_meta: Metadata for the test dataset.
        outputs_dir: Directory where the CSV files should be saved.
    """

    output_dataset(
        X=X_train,
        y=y_train,
        meta=train_meta,
        outputs_dir=outputs_dir,
        filename="train_set.csv",
    )

    output_dataset(
        X=X_test,
        y=y_test,
        meta=test_meta,
        outputs_dir=outputs_dir,
        filename="test_set.csv",
    )


def output_dataset(X, y, meta, outputs_dir, filename: str) -> None:
    """
    Save a dataset containing metadata, model input features, and target values.

    This can be used for training, test, or final model datasets.
    """
    if not meta.index.equals(X.index):
        raise ValueError("Metadata index does not match feature dataset index.")

    output_df = meta.copy()
    output_df = pd.concat([output_df, X.copy()], axis=1)
    output_df["Weekly_Sales_tplus1"] = y

    output_path = outputs_dir / filename
    output_df.to_csv(output_path, index=False)

    print(f"Saved dataset to CSV: {output_path.resolve()}")
    print(f"  {filename} → {output_df.shape[0]:,} rows")


def output_feature_importance(model, X_train, outputs_dir) -> None:
    """
    Save XGBoost feature importance values and a feature importance plot.

    The full feature importance table is saved as a CSV file, and a horizontal
    bar chart is saved as a PNG image in the xgb_outputs directory.

    Args:
        model: Trained XGBoost model.
        X_train: Training feature dataset used to fit the model.
        outputs_dir: Base output directory where feature importance files
            should be saved.
    """

    xgb_output_dir = outputs_dir / "xgb_outputs"
    xgb_output_dir.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    fscore = booster.get_score(importance_type="gain")  # raw feature importance
    name_map = {f"f{idx}": col for idx, col in enumerate(X_train.columns)}

    # Map to full column list, fill zeros for unused
    gain_map = {name_map.get(k, k): v for k, v in fscore.items()}
    imp_full = (
        pd.DataFrame({"feature": list(X_train.columns)})
        .assign(gain=lambda d: d["feature"].map(gain_map).fillna(0.0))
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    # Save to CSV
    imp_full.to_csv(xgb_output_dir / "feature_importance_gain_full.csv", index=False)

    # Plot
    plt.figure(figsize=(10, max(6, len(imp_full) * 0.3)))
    plt.barh(imp_full["feature"][::-1], imp_full["gain"][::-1])
    plt.xlabel("Gain (Average Loss Reduction)")
    plt.title("XGBoost Feature Importance (Gain) - All Features")
    plt.tight_layout()
    plt.savefig(xgb_output_dir / "feature_importance_all.png", dpi=150)
    plt.close()

    print(f"\nSaved full importance list and plot to: {xgb_output_dir.resolve()}")


def output_challenger_forecast_by_store(
    test: pd.DataFrame,
    predictions,
    outputs_dir,
) -> None:
    """
    Save the challenger predictions for the held-out test week.

    The test row's Date is the feature date, while Weekly_Sales_tplus1
    is the actual sales value for the following week.
    """
    forecast_output = test[
        [
            "Store",
            "Date",
            "Weekly_Sales",
            "Weekly_Sales_tplus1",
        ]
    ].copy()

    forecast_output["Date"] = pd.to_datetime(forecast_output["Date"])

    forecast_output["Next_Week_Date"] = (
        forecast_output["Date"] + pd.Timedelta(weeks=1)
    )

    forecast_output["Predicted_Weekly_Sales"] = predictions.astype(float)

    forecast_output = forecast_output.rename(
        columns={
            "Weekly_Sales_tplus1": "Actual_Next_Week_Sales",
        }
    )

    forecast_output = forecast_output[
        [
            "Store",
            "Date",
            "Next_Week_Date",
            "Weekly_Sales",
            "Actual_Next_Week_Sales",
            "Predicted_Weekly_Sales",
        ]
    ]

    forecast_output = forecast_output.sort_values("Store")

    output_path = outputs_dir / "challenger_forecast_by_store.csv"

    forecast_output.to_csv(
        output_path,
        index=False,
        date_format="%d-%m-%Y",
    )

    print(f"\nSaved challenger forecast to: {output_path.resolve()}")
    print(f"Forecast rows: {len(forecast_output):,}")
    print(
        f"Feature date: {forecast_output['Date'].min().date()} "
        f"| Prediction date: "
        f"{forecast_output['Next_Week_Date'].min().date()}"
    )
