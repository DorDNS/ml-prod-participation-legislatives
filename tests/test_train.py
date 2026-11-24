import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException

from ml import train as train_module
from ml.dataset import FEATURES, TARGET
from ml.train import (
    build_model,
    champion_comparison,
    persistence_mae,
    predict_turnout,
    read_dvc_md5,
)


def make_table(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    prev = rng.uniform(40, 80, n)
    return pd.DataFrame(
        {
            "prev_turnout": prev,
            "prev2_turnout": prev + rng.normal(0, 3, n),
            "registered": rng.integers(200, 3000, n).astype("float64"),
            "dept_code": rng.choice(["01", "13", "2A", "75"], n),
            "election_year": rng.choice([2017, 2022], n),
            "turnout": prev + rng.normal(0, 2, n),
        }
    )


def test_model_predicts_one_value_per_row_within_bounds():
    table = make_table()
    model = build_model(max_iter=20).fit(table[FEATURES], table[TARGET])
    predictions = predict_turnout(model, table)
    assert predictions.shape == (len(table),)
    assert predictions.min() >= 0 and predictions.max() <= 100


def test_unknown_department_does_not_break_prediction():
    table = make_table()
    model = build_model(max_iter=20).fit(table[FEATURES], table[TARGET])
    unseen = table.head(1).copy()
    unseen["dept_code"] = "99"
    assert np.isfinite(predict_turnout(model, unseen)).all()


def test_persistence_mae_is_the_gap_to_previous_turnout():
    evaluation = pd.DataFrame(
        {"prev_turnout": [50.0, 60.0], "turnout": [55.0, 58.0]}
    )
    assert persistence_mae(evaluation) == pytest.approx(3.5)


def test_read_dvc_md5(tmp_path):
    pointer = tmp_path / "table.parquet.dvc"
    pointer.write_text(
        "outs:\n- md5: 5885fb1158f684e5f163593e3344ab71\n  size: 3890923\n"
    )
    assert read_dvc_md5(pointer) == "5885fb1158f684e5f163593e3344ab71"


def test_read_dvc_md5_rejects_a_file_without_hash(tmp_path):
    pointer = tmp_path / "broken.dvc"
    pointer.write_text("outs: []\n")
    with pytest.raises(ValueError):
        read_dvc_md5(pointer)


class FakeModelVersion:
    def __init__(self):
        self.version = "1"
        self.tags = {"mae": "3.0", "dataset_version": "v1"}


class FakeClient:
    """Stands in for MlflowClient: has a champion, or none."""

    def __init__(self, champion=None):
        self.champion = champion

    def get_model_version_by_alias(self, name, alias):
        if self.champion is None:
            raise MlflowException("no champion")
        return self.champion


class ConstantModel:
    def predict(self, data):
        return np.full(len(data), 55.0)


def test_champion_comparison_is_none_without_champion():
    assert champion_comparison(FakeClient(), "model", make_table()) is None


def test_champion_comparison_measures_the_champion_on_the_given_rows(monkeypatch):
    monkeypatch.setattr(
        train_module.mlflow.pyfunc, "load_model", lambda uri: ConstantModel()
    )
    table = make_table().head(10)
    result = champion_comparison(FakeClient(FakeModelVersion()), "model", table)
    assert result["mae"] == pytest.approx(float((table[TARGET] - 55.0).abs().mean()))
    assert result["version"] == "1"
    assert result["dataset_version"] == "v1"
    assert result["mae_reference"] == pytest.approx(3.0)
