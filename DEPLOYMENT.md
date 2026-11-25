# Deployment

This document explains how the application is built, configured and deployed, and how it follows the [Twelve-Factor App](https://12factor.net/) principles.

## Workflows and triggers

| Trigger | File | Jobs |
|---|---|---|
| Pull request to `dev`, `staging` or `main` | `ci.yml` | `lint-and-test`: ruff and pytest<br>`docker-build-and-smoke`: builds the image, starts the container, checks `/health` |
| Push to `dev` (or manual run) | `publish-image.yml` | `publish`: builds the image and pushes it to ghcr.io, tagged with the commit SHA |
| Push to `staging` | `publish-image.yml` and `staging.yml` (run in parallel) | `publish` (as above)<br>`build-and-smoke`: builds the image, starts the container in GitHub Actions with the real `champion` model, runs the smoke test with `--expect-champion` |
| Push to `main` | `prod.yml` | `publish`: reuses `publish-image.yml`<br>`deploy`: calls the Render deploy hook with the image of the commit SHA<br>`verify`: smoke test on the production URL with `--expect-champion` (up to 300 s for the cold start) |
| Manual (`workflow_dispatch`), `operation = pipeline` | `ct.yml` | `validate`: checks the dataset version and its Git tag<br>`train`: pulls the data with DVC, trains, evaluates and registers the candidate<br>`decide`: applies the promotion rule (`ml/policy.py`)<br>`promote`: only if the decision is `promote` |
| Manual (`workflow_dispatch`), `operation = rollback` | `ct.yml` | `rollback`: moves the `champion` alias back to a previous version |

## Twelve-factor checklist

| # | Factor | How the project applies it |
|---|---|---|
| I | Codebase | One Git repository, with `feature/*` → `dev` → `staging` → `main` branches. Each environment runs a different branch of the same codebase. |
| II | Dependencies | Dependencies are declared in `requirements.txt` (runtime) and `requirements-dev.txt` (tests and lint) and installed in an isolated environment (venv locally, the Docker image in production). |
| III | Config | Configuration is read from environment variables, never written in the code. See [Configuration](#configuration) for the full list, where each variable is used and how it is stored. |
| IV | Backing services | The model registry (MLflow on DagsHub) and the data storage (DVC on Azure Blob Storage) are attached resources, reached through URLs and credentials from the configuration. |
| V | Build, release, run | Build: `publish-image.yml` builds the Docker image from the repository. Release: the image is tagged with the commit SHA and pushed to ghcr.io. Run: Render starts that exact image with its environment variables, triggered by the deploy hook in `prod.yml`. |
| VI | Processes | The application is stateless: no data or model is stored in the image. At startup it downloads the model with the `champion` alias from the MLflow registry. |
| VII | Port binding | The app is served by uvicorn on the port given by the `PORT` variable (default `8000`), as set in the `Dockerfile`. |
| VIII | Concurrency | The app runs as a single uvicorn process. It can be scaled out by running more containers, since it keeps no local state. |
| IX | Disposability | The container starts with `exec uvicorn`, so uvicorn receives the stop signal from Docker and shuts down cleanly. The model is loaded in a background thread: `/health` answers immediately, and `/ready` returns 503 until the model is loaded, then 200. |
| X | Dev/prod parity | Every environment builds its image from the same `Dockerfile`, with the same Python version (3.12) locally, in CI and in production. The image deployed to production is exactly the one published by `prod.yml` for the merge commit on `main`. |
| XI | Logs | Logs are written to stdout (`PYTHONUNBUFFERED=1`), so the hosting platform collects them. The log level is set with `LOG_LEVEL` (default `INFO`). The app does not manage log files. |
| XII | Admin processes | Training, promotion and rollback run as one-off jobs in `ct.yml`, never inside the web application. |

## Configuration

| Variable | Used by | Stored as | Purpose |
|---|---|---|---|
| `PORT` | Application (runtime) | Set by the hosting platform, default `8000` | Port uvicorn listens on |
| `LOG_LEVEL` | Application (runtime, optional) | Environment variable, default `INFO` | Log level of the application |
| `MODEL_NAME` | Application (runtime), `staging.yml`, `prod.yml`, `ct.yml` | Environment variable | Name of the model in the MLflow registry (`polling-station-turnout`) |
| `MODEL_ALIAS` | Application (runtime), `staging.yml`, `prod.yml`, `ct.yml` | Environment variable | Alias of the model to load (`champion`) |
| `MLFLOW_TRACKING_URI` | Application (runtime), `staging.yml`, `prod.yml`, `ct.yml` | GitHub variable | URL of the MLflow server on DagsHub |
| `MLFLOW_TRACKING_USERNAME` | Application (runtime), `staging.yml`, `prod.yml`, `ct.yml` | GitHub variable | DagsHub user for MLflow |
| `MLFLOW_TRACKING_PASSWORD` | Application (runtime), `staging.yml`, `prod.yml`, `ct.yml` | GitHub secret | DagsHub token for MLflow |
| `AZURE_SAS_TOKEN` | `ct.yml` only (job `train`, for `dvc pull`) | GitHub secret | Access to the DVC remote on Azure. Never used by the application. |
| `RENDER_DEPLOY_HOOK_URL` | `prod.yml` and `ct.yml` | GitHub secret | Render deploy hook, used to deploy a given image |
| `PROD_URL` | `prod.yml` and `ct.yml` | GitHub variable | Public URL of the production service, used by the smoke tests |

The runtime variables are also set in the Render service's environment settings. No secret is written in the code, the image or the repository.

## Changing the model: promotion and rollback

The API loads the model **only at startup**. Moving the `champion` alias in the MLflow registry does not change the model used by a running application. This is why `ct.yml` redeploys the service after every promotion or rollback.

After moving the alias (job `promote` or `rollback`), the workflow:

1. Calls the Render deploy hook with the image of the current commit (`ghcr.io/dordns/ml-prod-participation-legislatives:<sha>`), so the service restarts and loads the new `champion`.
2. Polls `/model-info` every 10 seconds, for up to 10 minutes, until it returns the expected version.
3. Runs the smoke test with `--expect-version <version>`.

A rollback is only accepted towards a version that has already been champion. `ml/promote.py` tags a version `approved=true` when it is promoted, and tags the champion it replaces too. A rollback is refused if the target version does not exist, is already the champion, or was never approved, so a candidate rejected by the promotion rule can never become champion through a rollback. The version must be given as a number in the `rollback_version` input.

### If the expected version is not served

If production does not serve the expected version within 10 minutes, the job fails. The procedure is then:

1. Check that the service still answers: `GET /ready` and `GET /model-info`.
2. Read the failed job log and the Events tab of the Render service.
3. Run `ct.yml` with `operation = rollback` and `rollback_version` set to the last approved version.

The same applies to a production deployment from `prod.yml`: if the new deployment does not become ready, the `verify` job fails. Do not rerun it in a loop: check `/ready` first, then read the `verify` log and the Render Events tab.

## Reproducing a deployment

1. **Hosting**: create a Render web service of type *Existing image* (Docker image) using `ghcr.io/dordns/ml-prod-participation-legislatives:<sha>`. The image is public, so no registry credentials are needed.
2. **Environment variables of the service**: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `MODEL_NAME`, `MODEL_ALIAS`, and optionally `LOG_LEVEL`.
3. **In GitHub** (Settings, Secrets and variables, Actions): add the secret `RENDER_DEPLOY_HOOK_URL` (the service's deploy hook, found in Render under Settings, Deploy Hook) and the variable `PROD_URL`.
4. **Deploy**: a push to `main` triggers `prod.yml`, which publishes the image of the commit, deploys it and verifies it.
5. **Production URL**: <https://turnout-api-g0tp.onrender.com> (HTTPS provided by Render). Endpoints: `GET /health`, `GET /ready`, `GET /model-info`, `POST /predict`, and the interactive documentation at `/docs`.
6. **Render free plan**: the service sleeps after 15 minutes without traffic, and the first request after sleep can take about a minute. The file system is ephemeral: nothing is stored locally, which is consistent with factor VI.
