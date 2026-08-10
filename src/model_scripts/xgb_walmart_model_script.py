import argparse
import os
import mlflow

from src.common.model_helpers import (
    validation
)
from src.common.model_helpers import config, data, encoding, features, metrics, mlflow_utils, model, outputs


# ----------------------------- Training -----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, required=True, help='Dataset for training')
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument(
        "--challenger_model",
        action="store_true",
        help="Train a challenger model for comparison with the production model.",
    )
    args = parser.parse_args()

    outputs_dir = data.get_outputs_dir()

    df = data.load_dataset(args.training_data)

    df_feat = features.build_features(df)

    # Chronological split
    if args.challenger_model:
        test_horizon_weeks = config.CHALLENGER_HORIZON_WEEKS
        print(f"Challenger model: using {test_horizon_weeks} week(s) for test set.")
    else:
        test_horizon_weeks = config.TEST_HORIZON_WEEKS
        print(f"Candidate model: using {test_horizon_weeks} weeks for test set.")

    train, test = data.chronological_split(df_feat, test_horizon_weeks)

    # Keep metadata for output BEFORE encoding
    train_meta = train[config.METADATA_COLUMNS].copy()
    test_meta = test[config.METADATA_COLUMNS].copy()

    # One-hot encode
    train_enc = encoding.prepare_encodings(train)
    test_enc = encoding.prepare_encodings(test)

    # Build the final list of columns that will be used as model input features.
    feature_cols = encoding.get_feature_columns(train_enc)
    required_cols = feature_cols + ["Weekly_Sales_tplus1"]
    print(f"Features: {len(required_cols):,}")
    print(required_cols)

    validation.validate_missing_values(train_enc, "training", required_cols)
    validation.validate_missing_values(test_enc, "test", required_cols)

    # Separate features and target
    X_train, y_train = train_enc[feature_cols].copy(), train_enc["Weekly_Sales_tplus1"].copy()
    X_test, y_test = test_enc[feature_cols].copy(), test_enc["Weekly_Sales_tplus1"].copy()
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")

    outputs.output_train_test_sets(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, train_meta=train_meta, test_meta=test_meta, outputs_dir=outputs_dir)

    mlflow_utils.configure_mlflow()
    with mlflow.start_run():

        # Log dataset/path & split info
        mlflow.log_param("training_data", os.path.abspath(args.training_data))
        mlflow.log_param("test_horizon_weeks", test_horizon_weeks)
        mlflow.log_param("challenger_model", args.challenger_model)
        mlflow.log_param("roll_windows", ",".join(map(str, config.ROLL_WINDOWS)))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_train_rows", int(len(X_train)))
        mlflow.log_param("n_test_rows", int(len(X_test)))

        # Train model
        xgb_model = model.create_xgb_model(args)

        # Log model hyperparameters
        mlflow.log_params(xgb_model.get_params())

        # Train the model on the training data
        xgb_model.fit(X_train, y_train)
        
        # In future version 
        # - train directly from the feature store
        # - Create the feature_retrieval_spec.yaml dynamically and add it as as artifact
        # spec_path = "feature_retrieval_spec/feature_retrieval_spec.yaml"
        # mlflow.log_artifact(
        #     local_path=spec_path,
        #     artifact_path="model"
        # )

        # Evaluate the model on the test data
        preds = xgb_model.predict(X_test)

        metrics.calculate_regression_metrics(y_test, preds)

        # mlflow.log_artifacts not used as there is a bug.
        outputs.output_feature_importance(xgb_model, X_train, outputs_dir)

        if args.challenger_model:
            outputs.output_challenger_forecast_by_store(
                test=test,
                predictions=preds,
                outputs_dir=outputs_dir,
            )
        
        # model_path = outputs_dir / "model.pkl"
        # joblib.dump(xgb_model, model_path)


if __name__ == "__main__":
    main()


