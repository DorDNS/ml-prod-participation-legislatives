"""FastAPI application serving the champion turnout model from the MLflow registry."""

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient

logger = logging.getLogger("turnout-api")

MODEL_NAME = os.environ.get("MODEL_NAME", "polling-station-turnout")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")


@dataclass
class ModelState:
    """What the API knows about the model it serves."""

    model: object | None = None
    name: str = MODEL_NAME
    alias: str = MODEL_ALIAS
    version: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


def load_champion() -> ModelState:
    """Resolve the alias to a concrete version, then load exactly that version."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    return ModelState(model=model, name=MODEL_NAME, alias=MODEL_ALIAS, version=str(version))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; stay alive but not ready if it fails."""
    try:
        app.state.model_state = load_champion()
        logger.info("Loaded %s version %s", MODEL_NAME, app.state.model_state.version)
    except Exception:
        logger.exception("Could not load the champion model; the API is not ready")
    yield


app = FastAPI(title="Polling station turnout", version="0.0.1", lifespan=lifespan)
# Empty state by default: tests can replace it with a fake model.
app.state.model_state = ModelState()


def get_loaded_state(request: Request) -> ModelState:
    """Return the model state, or answer 503 if no model is loaded."""
    state: ModelState = request.app.state.model_state
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, bool]:
    """Readiness probe: 200 only when the model is loaded, 503 otherwise."""
    get_loaded_state(request)
    return {"ready": True}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, str | None]:
    """Report which model version is actually loaded."""
    state = get_loaded_state(request)
    return {"model_name": state.name, "alias": state.alias, "version": state.version}