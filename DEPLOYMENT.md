# Deployment

This document explains how the application is built, configured and deployed, and how it follows the [Twelve-Factor App](https://12factor.net/) principles.

> **Status: first draft.** Deployment values (hosting URL, service settings) will be added once the Render service is created. Rows marked *Planned* describe what is not implemented yet.

## Workflows and triggers

| Trigger | Workflow | Jobs | Status |
|---|---|---|---|
| Pull request to `dev`, `staging` or `main` | `ci.yml` | `lint-and-test`: ruff and pytest<br>`docker-build-and-smoke`: builds the image, starts the container, checks `/health` | Implemented |
| Merge to `staging` | *To be defined* | Builds the image and validates a test container started by GitHub Actions | Planned |
| Merge to `main` | *To be defined* | Builds the image tagged with the commit SHA, deploys it, waits for `/ready`, runs a smoke test on the live URL | Planned |
| Manual (`workflow_dispatch`) with a data version as input | *To be defined* | Retrains a candidate model, compares it with the `champion`, promotes it or keeps the champion | Planned |
| Manual (`workflow_dispatch`) | *To be defined* | Rolls back the `champion` alias to a previous model version | Planned |

## Twelve-factor checklist

| # | Factor | How the project applies it |
|---|---|---|
| I | Codebase | One Git repository, with `feature/*` → `dev` → `staging` → `main` branches. Each environment runs a different branch of the same codebase. |
| II | Dependencies | Dependencies are declared in `requirements.txt` (runtime) and `requirements-dev.txt` (tests and lint) and installed in an isolated environment (venv locally, the Docker image in production). |
| III | Config | Configuration is read from environment variables, never written in the code. See [Configuration](#configuration) for the full list, where each variable is used and how it is stored. |
| IV | Backing services | The model registry (MLflow on DagsHub) and the data storage (DVC on Azure Blob Storage) are attached resources, reached through URLs and credentials from the configuration. |
| V | Build, release, run | Build: the Docker image is built from the repository. Release: the image is tagged with the commit SHA. Run: the platform starts that image with its environment variables. *(Release and deployment steps planned.)* |
| VI | Processes | The application is stateless: no data or model is stored in the image. At startup it downloads the model with the `champion` alias from the MLflow registry. |
| VII | Port binding | The app is served by uvicorn on the port given by the `PORT` variable (default `8000`), as set in the `Dockerfile`. |
| VIII | Concurrency | The app runs as a single uvicorn process. It can be scaled out by running more containers, since it keeps no local state. |
| IX | Disposability | The container starts with `exec uvicorn`, so uvicorn receives the stop signal from Docker and shuts down cleanly. The `/ready` endpoint reports when the model is loaded. |
| X | Dev/prod parity | The same Docker image is built in CI and deployed to production, with the same Python version (3.12) used locally and in CI. |
| XI | Logs | Logs are written to stdout (`PYTHONUNBUFFERED=1`), so the hosting platform collects them. The app does not manage log files. |
| XII | Admin processes | Training, retraining and rollback run as separate one-off workflows, not inside the web application. *(Planned.)* |

## Configuration

| Variable | Used by | Stored as | Purpose |
|---|---|---|---|
| `PORT` | Application (runtime) | Set by the hosting platform, default `8000` | Port uvicorn listens on |
| `MODEL_NAME` | Application (runtime) | Environment variable | Name of the model in the MLflow registry (`polling-station-turnout`) |
| `MODEL_ALIAS` | Application (runtime) | Environment variable | Alias of the model to load (`champion`) |
| `MLFLOW_TRACKING_URI` | Application (runtime) and CT workflow | GitHub variable | URL of the MLflow server on DagsHub |
| `MLFLOW_TRACKING_USERNAME` | Application (runtime) and CT workflow | GitHub variable | DagsHub user for MLflow |
| `MLFLOW_TRACKING_PASSWORD` | Application (runtime) and CT workflow | GitHub secret | DagsHub token for MLflow |
| `AZURE_SAS_TOKEN` | CT workflow only (DVC) | GitHub secret | Access to the DVC remote on Azure. Never used by the application. |

The runtime variables are also set in the hosting platform's environment settings. No secret is written in the code, the image or the repository.

## Changing the model: promotion and rollback

The API loads the model **only at startup**. Moving the `champion` alias in the MLflow registry (after a promotion or a rollback) does not change the model used by a running application. The service must be restarted to pick up the new version.

After every promotion or rollback:

1. Redeploy or restart the service.
2. Wait until `/ready` returns success, which means the new model is loaded.
3. Call `/model-info` and check that it returns the expected model version.

If the application does not become ready after a model change (for example, the new model fails to load), move the `champion` alias back to the previous version, restart the service, and check `/ready` and `/model-info` again.

## Reproducing a deployment

*To be completed once the Render service is created.*
