"""Create a small MLflow registry in a folder, with a test champion model.

It lets the CI start the real Docker image with a model and run the smoke test,
without access to the shared registry or to any secret. The model is trained on
synthetic data with the real pipeline of ml.train: it is NOT meant to be accurate.

Usage (from the repository root):

    python scripts/make_test_registry.py --path /tmp/ci-registry

Then start the application with:

    MLFLOW_ALLOW_FILE_STORE=true MLFLOW_TRACKING_URI=file:///tmp/ci-registry \
        uvicorn app.main:app

The registry holds one version, which is the champion.
"""

import argparse
import os
import sys
from pathlib import Path

# MLflow keeps the file store in maintenance mode: it must be allowed explicitly.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.dataset import FEATURES, TARGET
from ml.train import ALIAS, DEFAULT_MODEL_NAME, build_model

DEPARTMENTS = ["01", "13", "2A", "59", "69", "75"]


def make_synthetic_table(n: int = 600) -> pd.DataFrame:
    """Fake polling stations with the five real features."""
    rng = np.random.default_rng(42)
    prev = rng.uniform(40, 80, n)
    prev2 = prev + rng.normal(0, 5, n)
    table = pd.DataFrame(
        {
            "prev_turnout": prev,
            "prev2_turnout": prev2,
            "registered": rng.integers(200, 3000, n).astype("float64"),
            "dept_code": rng.choice(DEPARTMENTS, n).astype(object),
            "election_year": rng.choice([2017, 2022, 2024], n).astype("int64"),
        }
    )
    table[TARGET] = np.clip(0.7 * prev + 0.3 * prev2 + rng.normal(0, 2, n), 0, 100)
    return table


def create_registry(path: Path, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Register one model version with the champion alias; return its version."""
    path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(path.resolve().as_uri())
    table = make_synthetic_table()
    model = build_model().fit(table[FEATURES], table[TARGET])
    predictions = model.predict(table[FEATURES])

    mlflow.set_experiment("ci-test-model")
    with mlflow.start_run(run_name="ci-test-model"):
        mlflow.set_tag("purpose", "ci-test-model")
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(table[FEATURES], predictions),
            input_example=table[FEATURES].head(3),
            registered_model_name=model_name,
            # Same serialization as the real training: the slim image can load it.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(model_name, ALIAS, version)
    return str(version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", required=True, help="folder of the test registry")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args(argv)
    version = create_registry(Path(args.path), args.model_name)
    print(f"Test registry in {args.path}: {args.model_name} version {version}, alias {ALIAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
