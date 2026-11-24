"""Tests for readiness and model info. No network and no MLflow calls."""

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import ModelState, app


class FakeModel:
    """Stands in for the MLflow model: always predicts the same value."""

    def predict(self, data):
        return [55.0]


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


def load_fake_model(version: str = "7") -> None:
    app.state.model_state = ModelState(model=FakeModel(), version=version)


def test_health_is_up_without_model(client):
    assert client.get("/health").status_code == 200


def test_ready_returns_503_when_model_is_not_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 503


def test_model_info_returns_503_when_model_is_not_loaded(client):
    assert client.get("/model-info").status_code == 503


def test_ready_returns_200_when_model_is_loaded(client):
    load_fake_model()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_model_info_reports_the_loaded_version(client):
    load_fake_model(version="7")
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "polling-station-turnout",
        "alias": "champion",
        "version": "7",
    }


def test_startup_loads_the_model(monkeypatch):
    monkeypatch.setattr(
        main,
        "load_champion",
        lambda: ModelState(model=FakeModel(), version="3"),
    )
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/model-info").json()["version"] == "3"


def test_startup_survives_a_registry_failure(monkeypatch):
    def fail():
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(main, "load_champion", fail)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503