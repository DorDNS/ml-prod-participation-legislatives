"""FastAPI application serving the champion turnout model from the MLflow registry."""

import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient

from app.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger("turnout-api")

MODEL_NAME = os.environ.get("MODEL_NAME", "polling-station-turnout")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
FEATURE_COLUMNS = [
    "prev_turnout",
    "prev2_turnout",
    "registered",
    "dept_code",
    "election_year",
]

@dataclass
class ModelState:
    """What the API knows about the model it serves."""

    model: object | None = None
    name: str = MODEL_NAME
    alias: str = MODEL_ALIAS
    version: str | None = None
    run_id: str | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


def load_champion() -> ModelState:
    """Resolve the alias to a concrete version, then load exactly that version."""
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
    return ModelState(
        model=model,
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=str(model_version.version),
        run_id=model_version.run_id,
    )


def load_in_background(app: FastAPI) -> None:
    """Load the model without blocking the server; stay not ready on failure."""
    try:
        app.state.model_state = load_champion()
        logger.info("Loaded %s version %s", MODEL_NAME, app.state.model_state.version)
    except Exception:
        logger.exception("Could not load the champion model; the API is not ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start loading the model in a thread so /health answers immediately."""
    app.state.loader_thread = threading.Thread(
        target=load_in_background, args=(app,), daemon=True
    )
    app.state.loader_thread.start()
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
def ready(request: Request) -> JSONResponse:
    """Readiness probe: 200 only when the model is loaded, 503 otherwise."""
    if not request.app.state.model_state.loaded:
        return JSONResponse(status_code=503, content={"ready": False})
    return JSONResponse(content={"ready": True})


@app.get("/model-info")
def model_info(request: Request) -> dict[str, str | None]:
    """Report which model version is actually loaded."""
    state = get_loaded_state(request)
    return {
        "model_name": state.name,
        "alias": state.alias,
        "version": state.version,
        "run_id": state.run_id,
    }

@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Predict the turnout (in %) of a polling station."""
    state = get_loaded_state(request)
    row = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    row = row.astype({"registered": "float64", "election_year": "int64"})
    raw_prediction = float(np.asarray(state.model.predict(row)).ravel()[0])
    # A turnout is a percentage: keep it inside [0, 100]
    turnout_pct = min(100.0, max(0.0, raw_prediction))
    return PredictionResponse(turnout_pct=turnout_pct, model_version=state.version)