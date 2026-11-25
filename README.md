# ML in Production: Polling Station Turnout Prediction

This project predicts the turnout (percentage of registered voters who vote) of a polling station in metropolitan France for the first round of a legislative election.
Predictions are served by a FastAPI API, deployed online, which loads the model behind the `champion` alias of the MLflow registry. A prediction uses five inputs: the turnout at the previous two elections, the number of registered voters, the department code and the election year.
The focus is the MLOps pipeline around the model: versioned data (DVC on Azure), tracked experiments and model registry (MLflow on DagsHub), automated tests, CI/CD with GitHub Actions, continuous training, promotion and rollback.

**Production:** <https://turnout-api-g0tp.onrender.com> · interactive API documentation: <https://turnout-api-g0tp.onrender.com/docs>

The service is on Render's free plan: the first request after 15 minutes without traffic can take about a minute.

Open the production URL in a browser to try the model with a simple form (the interactive API documentation is on `/docs`).

## Try the API

Production URL: <https://turnout-api-g0tp.onrender.com> (a browser form on `/`, interactive documentation on `/docs`). The first request after a pause can take about a minute.

**Inputs of `POST /predict`:**

| Field | Meaning | Valid values |
|---|---|---|
| `prev_turnout` | Turnout of the previous first round, in % | 0 to 100 |
| `prev2_turnout` | Turnout of the first round before that, in % | 0 to 100 |
| `registered` | Registered voters of the polling station | greater than 0 |
| `dept_code` | Department code, for example `"75"` or `"2A"` | 1 to 3 characters |
| `election_year` | Year of the election to predict | 2000 to 2100 |

**Sample inputs.** Three real polling stations of the 2022 first round (`election_year` 2022; the previous turnouts are those of 2017 and 2012). The observed turnout is given for comparison only: the API does not receive it.

| Polling station | `prev_turnout` | `prev2_turnout` | `registered` | `dept_code` | Observed turnout |
|---|---|---|---|---|---|
| Paris (`75056_1308`) | 55.5 | 61.1 | 1496 | `75` | 59.0 % |
| Marseille (`13055_0703`) | 50.9 | 59.2 | 978 | `13` | 49.9 % |
| Corsica (`2A004_0020`) | 44.4 | 50.7 | 1016 | `2A` | 44.3 % |

```bash
URL=https://turnout-api-g0tp.onrender.com

# Which model version is served
curl "$URL/model-info"

# Paris
curl -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"prev_turnout": 55.5, "prev2_turnout": 61.1, "registered": 1496, "dept_code": "75", "election_year": 2022}'

# Marseille
curl -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"prev_turnout": 50.9, "prev2_turnout": 59.2, "registered": 978, "dept_code": "13", "election_year": 2022}'

# Corsica
curl -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"prev_turnout": 44.4, "prev2_turnout": 50.7, "registered": 1016, "dept_code": "2A", "election_year": 2022}'
```

A successful answer looks like `{"turnout_pct": <number between 0 and 100>, "model_version": "<version>"}`. An input out of range (for example `"prev_turnout": 120`) is refused with status 422.

## Documentation

- [`DEPLOYMENT.md`](DEPLOYMENT.md): workflows, twelve-factor checklist, configuration, promotion and rollback, how to reproduce a deployment
- [`ml/README_data.md`](ml/README_data.md): data sources, dataset versions and split
- [`ml/README_experiments.md`](ml/README_experiments.md): experiments, metrics and promotion thresholds

## Repository structure

```
.
├── app/                        # FastAPI application (API endpoints)
├── ml/
│   ├── dataset.py              # Builds the modeling table from the raw data
│   ├── train.py                # Trains, evaluates and registers a model in MLflow
│   ├── policy.py               # Promotion and retraining rules
│   ├── promote.py              # Moves the champion alias (promotion, rollback)
│   ├── README_data.md
│   └── README_experiments.md
├── scripts/
│   └── smoke_test.py           # Smoke test of a running instance of the API
├── data/
│   └── modeling_table.parquet.dvc   # DVC pointer (the data itself is on Azure)
├── tests/                      # Automated tests (pytest)
└── .github/workflows/          # GitHub Actions workflows (CI/CD and continuous training)
```

## Local setup

Requirements: Python 3.12, Git, and Docker (for building the image).

1. Clone the repository and switch to the `dev` branch:

   ```bash
   git clone https://github.com/DorDNS/ml-prod-participation-legislatives.git
   cd ml-prod-participation-legislatives
   git checkout dev
   ```

2. Create and activate a virtual environment:

   macOS / Linux:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the development dependencies:

   ```bash
   python -m pip install -r requirements-dev.txt
   ```

4. Run the tests and the linter:

   ```bash
   python -m pytest
   python -m ruff check .
   ```

Both commands must pass before pushing any change.

## Run the application locally with Docker

Build the image, then start it with your own MLflow credentials (never commit them):

```bash
docker build -t turnout-app .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=<your-mlflow-tracking-uri> \
  -e MLFLOW_TRACKING_USERNAME=<your-dagshub-username> \
  -e MLFLOW_TRACKING_PASSWORD=<your-dagshub-token> \
  -e MODEL_NAME=polling-station-turnout \
  -e MODEL_ALIAS=champion \
  turnout-app
```

On Windows PowerShell, replace the `\` line continuations with backticks (`` ` ``) or write the command on one line.

Then open <http://localhost:8000/docs>. `/health` answers immediately, and `/ready` returns 200 once the model is loaded.

## Branch workflow

```
feature/* ──PR──> dev ──PR──> staging ──PR──> main
```

- All work is done on `feature/<topic>` branches created from `dev`.
- `dev`, `staging` and `main` are protected: direct pushes, force pushes and deletions are blocked.
- Every change goes through a pull request reviewed and approved by another team member, with a green CI.
- Changes are promoted with a pull request from `dev` to `staging`, then from `staging` to `main`. Promotions are merged with a **merge commit** (no squash, no rebase), so the three branches keep the same history.
- Commit messages and pull request titles follow the convention `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, and everything is written in English.

## Workflows

| Trigger | File | What it does |
|---|---|---|
| Pull request to `dev`, `staging` or `main` | `ci.yml` | Lint, tests, Docker build and smoke test of the container |
| Push to `dev` | `publish-image.yml` | Publishes the image to ghcr.io, tagged with the commit SHA |
| Push to `staging` | `publish-image.yml`, `staging.yml` | Publishes the image, and runs the container with the real `champion` model and a smoke test |
| Push to `main` | `prod.yml` | Publishes the image, deploys it to Render and verifies production |
| Manual | `ct.yml` | Continuous training (`pipeline`) or `rollback` |

See [`DEPLOYMENT.md`](DEPLOYMENT.md#workflows-and-triggers) for the jobs of each workflow.

## Continuous training

The `Continuous training` workflow (`ct.yml`) is started by hand:

1. Open the **Actions** tab, select **Continuous training**, click **Run workflow**.
2. Choose the branch **`main`**, then fill in the inputs:

| Input | Role |
|---|---|
| `operation` | `pipeline` (default): validate the data version, train a candidate, apply the promotion rule, and promote it only if the rule says so. `rollback`: move the `champion` alias back to a previous version. |
| `dataset_version` | Pipeline only. Data version to train on, such as `v2` (default). The Git tag `data-<version>` must exist. |
| `rollback_version` | Rollback only. Model version number to restore. It must have been champion before. |

After a promotion or a rollback, the workflow redeploys the service and checks that production serves the expected version. See [`DEPLOYMENT.md`](DEPLOYMENT.md#changing-the-model-promotion-and-rollback).
