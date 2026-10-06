# tests/test_model.py

from types import SimpleNamespace
from xgboost import XGBRegressor
from  walmart_ml.common.model_helpers import model


def test_create_xgb_model_returns_xgb_regressor(mocker):
    # Purpose: Check that create_xgb_model returns an XGBRegressor object and 
    # create_xgb_model correctly applies args, config values, and fixed XGBoost settings.
    mocker.patch("walmart_ml.common.model_helpers.config.RANDOM_STATE", 123)

    args = SimpleNamespace(
        n_estimators=600,
        eta=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
    )

    xgb_model = model.create_xgb_model(args)

    assert isinstance(xgb_model, XGBRegressor)

    params = xgb_model.get_params()

    assert params["n_estimators"] == 600
    assert params["learning_rate"] == 0.05
    assert params["max_depth"] == 8
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["reg_alpha"] == 0.1
    assert params["reg_lambda"] == 2.0
    assert params["random_state"] == 123
    assert params["n_jobs"] == -1
    assert params["tree_method"] == "hist"