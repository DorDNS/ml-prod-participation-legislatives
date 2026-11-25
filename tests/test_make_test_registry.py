import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from ml.train import ALIAS, DEFAULT_MODEL_NAME
from scripts.make_test_registry import create_registry, main


@pytest.fixture(autouse=True)
def restore_tracking_uri():
    """create_registry points MLflow to a temporary folder: undo it after the test."""
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def test_registry_has_one_version_with_the_champion_alias(tmp_path):
    version = create_registry(tmp_path / "registry")
    assert version == "1"
    client = MlflowClient()
    champion = client.get_model_version_by_alias(DEFAULT_MODEL_NAME, ALIAS)
    assert str(champion.version) == "1"
    assert len(client.search_model_versions(f"name='{DEFAULT_MODEL_NAME}'")) == 1


@pytest.mark.parametrize("dept_code", ["75", "2A", "99"])
def test_champion_predicts_on_the_api_contract(tmp_path, dept_code):
    # "99" is a department never seen in training: it must not break the model.
    create_registry(tmp_path / "registry")
    model = mlflow.pyfunc.load_model(f"models:/{DEFAULT_MODEL_NAME}@{ALIAS}")
    row = pd.DataFrame(
        [
            {
                "prev_turnout": 60.0,
                "prev2_turnout": 62.0,
                "registered": 1200.0,
                "dept_code": dept_code,
                "election_year": 2024,
            }
        ]
    )
    prediction = np.asarray(model.predict(row)).ravel()
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])


def test_cli_creates_a_registry_in_the_given_folder(tmp_path, capsys):
    folder = tmp_path / "registry"
    assert main(["--path", str(folder)]) == 0
    assert folder.exists()
    assert "version 1, alias champion" in capsys.readouterr().out


def test_custom_model_name(tmp_path):
    create_registry(tmp_path / "registry", model_name="other-model")
    champion = MlflowClient().get_model_version_by_alias("other-model", ALIAS)
    assert str(champion.version) == "1"
