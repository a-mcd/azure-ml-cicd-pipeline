"""Unit tests for shared Walmart scoring utilities."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.common.scoring_helpers import scoring_utils


# ---------------------------------------------------------------------------
# get_required_environment_variable
# ---------------------------------------------------------------------------


def test_get_required_environment_variable_returns_trimmed_value(
    monkeypatch,
):
    monkeypatch.setenv("TEST_VARIABLE", "  test-value  ")

    result = scoring_utils.get_required_environment_variable(
        "TEST_VARIABLE",
    )

    assert result == "test-value"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_get_required_environment_variable_raises_for_missing_or_empty_value(
    monkeypatch,
    value,
):
    if value is None:
        monkeypatch.delenv("TEST_VARIABLE", raising=False)
    else:
        monkeypatch.setenv("TEST_VARIABLE", value)

    with pytest.raises(
        RuntimeError,
        match=r"Required environment variable 'TEST_VARIABLE' is not set\.",
    ):
        scoring_utils.get_required_environment_variable("TEST_VARIABLE")


# ---------------------------------------------------------------------------
# find_model_file
# ---------------------------------------------------------------------------


def test_find_model_file_returns_expected_model_subdirectory_file(tmp_path):
    expected_path = tmp_path / "model" / "model.xgb"
    expected_path.parent.mkdir()
    expected_path.write_bytes(b"model-data")

    # A second match should not matter when the conventional path exists.
    other_path = tmp_path / "other" / "model.xgb"
    other_path.parent.mkdir()
    other_path.write_bytes(b"other-model")

    result = scoring_utils.find_model_file(tmp_path)

    assert result == expected_path


def test_find_model_file_finds_single_nested_model(tmp_path):
    model_path = tmp_path / "artifacts" / "outputs" / "custom.xgb"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model-data")

    result = scoring_utils.find_model_file(
        tmp_path,
        filename="custom.xgb",
    )

    assert result == model_path


def test_find_model_file_raises_when_model_does_not_exist(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r"Could not find 'model\.xgb' beneath",
    ):
        scoring_utils.find_model_file(tmp_path)


def test_find_model_file_raises_when_multiple_models_exist(tmp_path):
    first_path = tmp_path / "first" / "model.xgb"
    second_path = tmp_path / "second" / "model.xgb"

    first_path.parent.mkdir()
    second_path.parent.mkdir()

    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    with pytest.raises(
        RuntimeError,
        match=r"Multiple 'model\.xgb' files were found beneath",
    ):
        scoring_utils.find_model_file(tmp_path)


# ---------------------------------------------------------------------------
# download_feature_csv
# ---------------------------------------------------------------------------


def test_download_feature_csv_copies_remote_file_to_local_path(tmp_path):
    remote_path = tmp_path / "remote" / "features.csv"
    remote_path.parent.mkdir()
    remote_path.write_bytes(b"Store,Date\n1,01-01-2024\n")

    local_path = tmp_path / "worker" / "data" / "features.csv"

    result = scoring_utils.download_feature_csv(
        str(remote_path),
        local_path,
    )

    assert result == local_path
    assert local_path.read_bytes() == remote_path.read_bytes()


def test_download_feature_csv_creates_parent_directories(tmp_path):
    remote_path = tmp_path / "features.csv"
    remote_path.write_bytes(b"Store,Date\n1,01-01-2024\n")

    local_path = tmp_path / "nested" / "worker" / "features.csv"

    scoring_utils.download_feature_csv(
        str(remote_path),
        local_path,
    )

    assert local_path.is_file()


def test_download_feature_csv_raises_when_downloaded_file_is_empty(tmp_path):
    remote_path = tmp_path / "empty.csv"
    remote_path.write_bytes(b"")

    local_path = tmp_path / "downloaded" / "empty.csv"

    with pytest.raises(
        RuntimeError,
        match=r"The downloaded CSV is empty:",
    ):
        scoring_utils.download_feature_csv(
            str(remote_path),
            local_path,
        )


# ---------------------------------------------------------------------------
# download_data_asset_csv
# ---------------------------------------------------------------------------


def test_download_data_asset_csv_resolves_asset_and_downloads_file(
    mocker,
    tmp_path,
):
    credential = mocker.sentinel.credential
    mocker.patch.object(
        scoring_utils,
        "ManagedIdentityCredential",
        return_value=credential,
    )

    data_asset = SimpleNamespace(
        name="walmart_materialized_features",
        version="7",
        type="uri_file",
        path="azureml://datastores/test/paths/features.csv",
    )

    mock_ml_client = mocker.MagicMock()
    mock_ml_client.data.get.return_value = data_asset

    mock_ml_client_class = mocker.patch.object(
        scoring_utils,
        "MLClient",
        return_value=mock_ml_client,
    )

    local_path = tmp_path / "features.csv"
    mock_download = mocker.patch.object(
        scoring_utils,
        "download_feature_csv",
        return_value=local_path,
    )

    result = scoring_utils.download_data_asset_csv(
        subscription_id="subscription-id",
        resource_group_name="test-rg",
        workspace_name="test-workspace",
        data_asset_name="walmart_materialized_features",
        data_asset_version="7",
        local_path=local_path,
    )

    assert result == local_path

    mock_ml_client_class.assert_called_once_with(
        credential=credential,
        subscription_id="subscription-id",
        resource_group_name="test-rg",
        workspace_name="test-workspace",
    )

    mock_ml_client.data.get.assert_called_once_with(
        name="walmart_materialized_features",
        version="7",
    )

    mock_download.assert_called_once_with(
        data_asset.path,
        local_path,
    )


def test_download_data_asset_csv_raises_when_asset_has_no_path(
    mocker,
    tmp_path,
):
    mock_ml_client = mocker.MagicMock()
    mock_ml_client.data.get.return_value = SimpleNamespace(
        name="feature-data",
        version="1",
        type="uri_file",
        path=None,
    )

    mocker.patch.object(
        scoring_utils,
        "ManagedIdentityCredential",
    )
    mocker.patch.object(
        scoring_utils,
        "MLClient",
        return_value=mock_ml_client,
    )

    with pytest.raises(
        RuntimeError,
        match=r"Data asset feature-data:1 does not contain a storage path\.",
    ):
        scoring_utils.download_data_asset_csv(
            subscription_id="subscription-id",
            resource_group_name="test-rg",
            workspace_name="test-workspace",
            data_asset_name="feature-data",
            data_asset_version="1",
            local_path=tmp_path / "features.csv",
        )


@pytest.mark.parametrize(
    "asset_type",
    [
        "uri_folder",
        "URI_FOLDER",
        "custom-folder-type",
    ],
)
def test_download_data_asset_csv_rejects_folder_assets(
    mocker,
    tmp_path,
    asset_type,
):
    mock_ml_client = mocker.MagicMock()
    mock_ml_client.data.get.return_value = SimpleNamespace(
        name="feature-data",
        version="1",
        type=asset_type,
        path="azureml://datastores/test/paths/features",
    )

    mocker.patch.object(
        scoring_utils,
        "ManagedIdentityCredential",
    )
    mocker.patch.object(
        scoring_utils,
        "MLClient",
        return_value=mock_ml_client,
    )

    with pytest.raises(
        ValueError,
        match=r"this downloader requires a URI-file asset",
    ):
        scoring_utils.download_data_asset_csv(
            subscription_id="subscription-id",
            resource_group_name="test-rg",
            workspace_name="test-workspace",
            data_asset_name="feature-data",
            data_asset_version="1",
            local_path=tmp_path / "features.csv",
        )

from pathlib import Path


def test_download_feature_csv_raises_when_local_file_was_not_created(
    mocker,
):
    # Purpose: Check that a completed copy operation that does not create the
    # expected local file raises a clear error.
    local_path = mocker.MagicMock(spec=Path)
    local_path.is_file.return_value = False

    remote_context = mocker.MagicMock()
    remote_context.__enter__.return_value = mocker.sentinel.remote_file

    local_context = mocker.MagicMock()
    local_context.__enter__.return_value = mocker.sentinel.local_file
    local_path.open.return_value = local_context

    mocker.patch.object(
        scoring_utils.fsspec,
        "open",
        return_value=remote_context,
    )
    mocker.patch.object(
        scoring_utils.shutil,
        "copyfileobj",
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"The CSV was not downloaded to",
    ):
        scoring_utils.download_feature_csv(
            "azureml://datastores/test/paths/features.csv",
            local_path,
        )

    local_path.parent.mkdir.assert_called_once_with(
        parents=True,
        exist_ok=True,
    )
    local_path.open.assert_called_once_with(mode="wb")
    local_path.is_file.assert_called_once_with()

# ---------------------------------------------------------------------------
# load_feature_data
# ---------------------------------------------------------------------------


def test_load_feature_data_normalizes_sorts_and_returns_expected_data(
    tmp_path,
):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": ["2", "1", "1"],
        "Date": [
            "08-01-2024",
            "08-01-2024",
            "01-01-2024",
        ],
        "Temperature": [20.0, 11.0, 10.0],
    }).to_csv(csv_path, index=False)

    result = scoring_utils.load_feature_data(csv_path)

    assert result["Store"].tolist() == [1, 1, 2]
    assert str(result["Store"].dtype) == "int64"

    assert result["Date"].tolist() == list(pd.to_datetime([
        "2024-01-01",
        "2024-01-08",
        "2024-01-08",
    ]))

    assert result["Temperature"].tolist() == [10.0, 11.0, 20.0]
    assert result.index.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("columns", "expected_message"),
    [
        (
            {
                "Date": ["01-01-2024"],
                "Temperature": [10.0],
            },
            "Store",
        ),
        (
            {
                "Store": [1],
                "Temperature": [10.0],
            },
            "Date",
        ),
        (
            {
                "Temperature": [10.0],
            },
            "Date, Store",
        ),
    ],
)
def test_load_feature_data_raises_when_required_columns_are_missing(
    tmp_path,
    columns,
    expected_message,
):
    csv_path = tmp_path / "features.csv"
    pd.DataFrame(columns).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match=(
            r"Engineered-feature CSV is missing required columns: "
            + expected_message
        ),
    ):
        scoring_utils.load_feature_data(csv_path)


def test_load_feature_data_raises_when_store_contains_null(tmp_path):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": [1, None],
        "Date": ["01-01-2024", "08-01-2024"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"Store contains null values\.",
    ):
        scoring_utils.load_feature_data(csv_path)


def test_load_feature_data_raises_when_store_is_not_numeric(tmp_path):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": ["one"],
        "Date": ["01-01-2024"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"Unable to parse string|Unable to parse",
    ):
        scoring_utils.load_feature_data(csv_path)


def test_load_feature_data_raises_when_store_is_fractional(tmp_path):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": [1.5],
        "Date": ["01-01-2024"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"Store must contain integer IDs\.",
    ):
        scoring_utils.load_feature_data(csv_path)


def test_load_feature_data_raises_when_date_format_is_invalid(tmp_path):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": [1],
        "Date": ["2024-01-01"],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        scoring_utils.load_feature_data(csv_path)


def test_load_feature_data_raises_for_duplicate_store_and_date_rows(
    tmp_path,
):
    csv_path = tmp_path / "features.csv"

    pd.DataFrame({
        "Store": [1, 1, 2],
        "Date": [
            "01-01-2024",
            "01-01-2024",
            "01-01-2024",
        ],
        "Temperature": [10.0, 11.0, 20.0],
    }).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match=(
            r"The engineered-feature CSV contains duplicate "
            r"Store/Date rows\."
        ),
    ) as error:
        scoring_utils.load_feature_data(csv_path)

    assert "'Store': 1" in str(error.value)
    assert "'Date': '01-01-2024'" in str(error.value)


# ---------------------------------------------------------------------------
# load_xgboost_model
# ---------------------------------------------------------------------------


def test_load_xgboost_model_finds_and_loads_model(mocker, tmp_path):
    model_path = tmp_path / "model" / "model.xgb"

    mock_find_model = mocker.patch.object(
        scoring_utils,
        "find_model_file",
        return_value=model_path,
    )

    mock_booster = mocker.MagicMock()
    mock_booster_class = mocker.patch.object(
        scoring_utils.xgb,
        "Booster",
        return_value=mock_booster,
    )

    result = scoring_utils.load_xgboost_model(tmp_path)

    assert result is mock_booster
    mock_find_model.assert_called_once_with(tmp_path, "model.xgb")
    mock_booster_class.assert_called_once_with()
    mock_booster.load_model.assert_called_once_with(model_path)


def test_load_xgboost_model_uses_custom_filename(mocker, tmp_path):
    model_path = tmp_path / "artifacts" / "challenger.xgb"

    mock_find_model = mocker.patch.object(
        scoring_utils,
        "find_model_file",
        return_value=model_path,
    )

    mock_booster = mocker.MagicMock()
    mocker.patch.object(
        scoring_utils.xgb,
        "Booster",
        return_value=mock_booster,
    )

    scoring_utils.load_xgboost_model(
        tmp_path,
        filename="challenger.xgb",
    )

    mock_find_model.assert_called_once_with(
        tmp_path,
        "challenger.xgb",
    )
    mock_booster.load_model.assert_called_once_with(model_path)