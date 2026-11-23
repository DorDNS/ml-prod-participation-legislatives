"""Register a throwaway placeholder model in the MLflow registry.

It lets the API be developed before the real model exists. The model is NOT
meant to be accurate: it only follows the draft contract of ml/README_data.md
(4 input columns, 1 output: turnout in percent).

Usage (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD must be set):

    python ml/register_placeholder_model.py
"""

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_NAME = "polling-station-turnout"
ALIAS = "champion"
EXPERIMENT = "placeholder-model"

NUMERIC_COLUMNS = ["prev_turnout", "prev2_turnout", "registered"]
CATEGORICAL_COLUMNS = ["dept_code"]
DEPARTMENTS = ["01", "13", "2A", "59", "69", "75"]


def make_synthetic_data(n: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """Generate fake polling stations: turnout depends on the two lags."""
    rng = np.random.default_rng(42)
    prev = rng.uniform(40, 80, n)
    prev2 = prev + rng.normal(0, 5, n)
    features = pd.DataFrame(
        {
            "prev_turnout": prev,
            "prev2_turnout": prev2,
            "registered": rng.integers(200, 3000, n).astype("float64"),
            "dept_code": rng.choice(DEPARTMENTS, n).astype(object),
        }
    )
    target = pd.Series(
        np.clip(0.7 * prev + 0.3 * prev2 + rng.normal(0, 2, n), 0, 100),
        name="turnout",
    )
    return features, target


def build_model() -> Pipeline:
    """Numeric columns pass through, the department is one-hot encoded."""
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]
    )


def main() -> None:
    features, target = make_synthetic_data()
    model = build_model().fit(features, target)
    signature = infer_signature(features, model.predict(features))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="placeholder"):
        mlflow.set_tag("placeholder", "true")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=features.head(3),
            registered_model_name=MODEL_NAME,
            # cloudpickle keeps the slim API image free of extra dependencies.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    client = MlflowClient()
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "placeholder", "true")
    print(f"Registered {MODEL_NAME} version {version} with alias '{ALIAS}'")

    # Check: reload the model the same way the API will do it.
    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    print("Predictions on 2 rows:", loaded.predict(features.head(2)))
    current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    print("Alias points to version:", current.version)


if __name__ == "__main__":
    main()
