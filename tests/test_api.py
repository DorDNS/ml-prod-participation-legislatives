"""Tests for readiness and model info. No network and no MLflow calls."""

import threading

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import ModelState, app


class FakeModel:
    """Stands in for the MLflow model: returns a fixed value and records its input."""

    def __init__(self, value: float = 55.0):
        self.value = value
        self.last_input = None

    def predict(self, data):
        self.last_input = data
        return [self.value]


@pytest.fixture(autouse=True)
def reset_state():
    """Start every test without a model, and restore the state afterwards."""
    original = app.state.model_state
    app.state.model_state = ModelState()
    yield
    app.state.model_state = original


@pytest.fixture
def client():
    # Not used as a context manager: startup does not run, MLflow is never called.
    return TestClient(app)


def load_fake_model(
    version: str = "7", run_id: str = "run-abc", value: float = 55.0
) -> FakeModel:
    model = FakeModel(value)
    app.state.model_state = ModelState(model=model, version=version, run_id=run_id)
    return model


def test_health_is_up_without_model(client):
    assert client.get("/health").status_code == 200


def test_ready_returns_503_when_model_is_not_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"ready": False}


def test_model_info_returns_503_when_model_is_not_loaded(client):
    assert client.get("/model-info").status_code == 503


def test_ready_returns_200_when_model_is_loaded(client):
    load_fake_model()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_model_info_reports_the_loaded_version(client):
    load_fake_model(version="7", run_id="run-abc")
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "polling-station-turnout",
        "alias": "champion",
        "version": "7",
        "run_id": "run-abc",
    }


def test_startup_loads_the_model(monkeypatch):
    monkeypatch.setattr(
        main,
        "load_champion",
        lambda: ModelState(model=FakeModel(), version="3"),
    )
    with TestClient(app) as client:
        app.state.loader_thread.join(timeout=5)
        assert client.get("/ready").status_code == 200
        assert client.get("/model-info").json()["version"] == "3"


def test_startup_survives_a_registry_failure(monkeypatch):
    def fail():
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(main, "load_champion", fail)
    with TestClient(app) as client:
        app.state.loader_thread.join(timeout=5)
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503


def test_health_answers_while_the_model_is_still_loading(monkeypatch):
    release = threading.Event()

    def slow_load():
        release.wait(timeout=5)
        return ModelState(model=FakeModel(), version="3")

    monkeypatch.setattr(main, "load_champion", slow_load)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        release.set()
        app.state.loader_thread.join(timeout=5)
        assert client.get("/ready").status_code == 200

VALID_PAYLOAD = {
    "prev_turnout": 62.4,
    "prev2_turnout": 58.1,
    "registered": 1200,
    "dept_code": "75",
    "election_year": 2024,
}


def test_predict_returns_turnout_and_model_version(client):
    load_fake_model(version="7", value=61.5)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"turnout_pct": 61.5, "model_version": "7"}


def test_predict_version_matches_model_info(client):
    load_fake_model(version="7")
    predicted = client.post("/predict", json=VALID_PAYLOAD).json()["model_version"]
    assert predicted == client.get("/model-info").json()["version"]


def test_predict_builds_the_expected_dataframe(client):
    model = load_fake_model()
    client.post("/predict", json=VALID_PAYLOAD)
    row = model.last_input
    assert list(row.columns) == [
        "prev_turnout",
        "prev2_turnout",
        "registered",
        "dept_code",
        "election_year",
    ]
    assert len(row) == 1
    assert row["registered"].dtype == "float64"
    assert row["election_year"].dtype == "int64"
    assert row["dept_code"].iloc[0] == "75"


@pytest.mark.parametrize("dept_code", ["01", "75", "2A", "2B", "999"])
def test_predict_accepts_any_department_code(client, dept_code):
    load_fake_model()
    response = client.post("/predict", json={**VALID_PAYLOAD, "dept_code": dept_code})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"prev_turnout": 120},
        {"prev_turnout": -5},
        {"prev2_turnout": 100.5},
        {"registered": 0},
        {"registered": -10},
        {"dept_code": ""},
        {"dept_code": "ABCD"},
        {"election_year": 1999},
        {"election_year": 2101},
        {"election_year": 2024.5},
        {"prev_turnout": "abc"},
        {"election_year": "twenty"},
    ],
)
def test_predict_rejects_invalid_values(client, changes):
    load_fake_model()
    response = client.post("/predict", json={**VALID_PAYLOAD, **changes})
    assert response.status_code == 422


@pytest.mark.parametrize("missing", list(VALID_PAYLOAD))
def test_predict_rejects_a_missing_field(client, missing):
    load_fake_model()
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != missing}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_returns_503_when_model_is_not_loaded(client):
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503


@pytest.mark.parametrize(
    "raw, expected", [(120.0, 100.0), (-3.0, 0.0), (0.0, 0.0), (100.0, 100.0)]
)
def test_predict_clips_the_output_to_0_100(client, raw, expected):
    load_fake_model(value=raw)
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.json()["turnout_pct"] == expected