from xgboost import XGBRegressor
from src.common.model_helpers import config

def create_xgb_model(args) -> XGBRegressor:
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
