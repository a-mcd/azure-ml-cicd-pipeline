"""XGBoost model definition for weekly sales forecasting."""

import argparse

from xgboost import XGBRegressor
from walmart_ml.training import config

def create_xgb_model(args: argparse.Namespace) -> XGBRegressor:
    """Create an untrained XGBoost regressor from the command-line arguments.

    Args:
        args: Parsed training arguments. Must include ``n_estimators``, ``eta``,
            ``max_depth``, ``subsample``, ``colsample_bytree``, ``reg_alpha`` and
            ``reg_lambda``.

    Returns:
        XGBRegressor: Model configured with a fixed random state and the
        histogram tree method.
    """
    return XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.eta,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
