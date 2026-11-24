# ML in Production: Polling Station Turnout Prediction

This project predicts the turnout of each polling station in metropolitan France for the first round of the next legislative election.
Predictions are served by a FastAPI application, packaged with Docker and displayed on a map of France where users can search for a municipality and choose a national turnout scenario.
The focus is the MLOps pipeline around the model: versioned data (DVC on Azure), tracked experiments and model registry (MLflow on DagsHub), automated tests, CI/CD with GitHub Actions, retraining, promotion and rollback.

## Repository structure

```
.
├── app/                 # FastAPI application (API endpoints and demo page)
├── ml/                  # Data preparation, training and model registration
├── tests/               # Automated tests (pytest)
└── .github/workflows/   # GitHub Actions workflows (CI/CD)
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

## Branch workflow

```
feature/* ──PR──> dev ──PR──> staging ──PR──> main
```

- All work is done on `feature/<topic>` branches created from `dev`.
- `dev`, `staging` and `main` are protected: direct pushes, force pushes and deletions are blocked.
- Every change goes through a pull request reviewed and approved by another team member.
- Changes are promoted with a pull request from `dev` to `staging`, then from `staging` to `main`.
- Commit messages and pull request titles follow the convention `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, and everything is written in English.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| CI | Pull request | Runs the linter (ruff) and the tests (pytest) |

More workflows (staging and production deployment, retraining, rollback) will be added as the project progresses.
