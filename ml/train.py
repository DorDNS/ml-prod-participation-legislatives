"""Train the turnout model, log the run in MLflow and register the model.

The model predicts the turnout (%) of a polling station at a legislative first
round from the five features of ml.dataset.FEATURES. Every run logs the Git
SHA, the dataset version, the DVC hash, the hyperparameters, the metrics, the
model and an evaluation report.

Metric: mean absolute error (MAE) in turnout points, lower is better. The
reference is the persistence baseline ("same turnout as the previous election")
computed on the very same evaluation rows.

Usage (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD must be set):

    # Initial release: register the model and give it the alias "champion"
    python -m ml.train --dataset-version v1 --bootstrap

    # Candidate: register the model without touching the alias
    python -m ml.train --dataset-version v2 --register --output candidate.json
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.dataset import FEATURES, HOLDOUT_SHARE, TARGET, split_train_eval

CATEGORICAL = ["dept_code"]
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
DEFAULT_MODEL_NAME = "polling-station-turnout"
ALIAS = "champion"
EXPERIMENT = "turnout-regression"


def build_model(max_iter: int = 300, learning_rate: float = 0.05) -> Pipeline:
    """Department encoded inside the pipeline, then gradient boosting.

    The API sends plain strings, so the encoding must live in the model.
    A department never seen at training time becomes a missing value.
    """
    preprocess = ColumnTransformer(
        [
            (
                "dept",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
                CATEGORICAL,
            ),
            ("num", "passthrough", NUMERIC),
        ]
    )
    regressor = HistGradientBoostingRegressor(
        categorical_features=[0],  # the encoded department is the first column
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def predict_turnout(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predicted turnout in percent, bounded to [0, 100] like the API does."""
    return np.clip(model.predict(features[FEATURES]), 0, 100)


def persistence_mae(evaluation: pd.DataFrame) -> float:
    """MAE of the baseline that repeats the previous turnout."""
    return float(mean_absolute_error(evaluation[TARGET], evaluation["prev_turnout"]))


def chronological_metrics(table: pd.DataFrame, params: dict) -> dict[str, float]:
    """Complementary check: train on earlier elections, test on the latest one."""
    latest = table["election_year"].max()
    past = table[table["election_year"] < latest]
    test = table[table["election_year"] == latest]
    model = build_model(**params).fit(past[FEATURES], past[TARGET])
    return {
        "chrono_mae": float(
            mean_absolute_error(test[TARGET], predict_turnout(model, test))
        ),
        "chrono_mae_persistence": persistence_mae(test),
    }


def read_dvc_md5(dvc_file: Path) -> str:
    """Hash of the dataset, read from its DVC pointer file."""
    match = re.search(r"md5:\s*([0-9a-f]{32})", dvc_file.read_text())
    if match is None:
        raise ValueError(f"No md5 hash found in {dvc_file}")
    return match.group(1)


def git_state() -> tuple[str, bool]:
    """Current commit SHA and whether tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            # The DVC pointer is excluded: it changes to select a dataset version,
            # which is already logged on its own (dataset_version, dvc_md5).
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!*.dvc"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return sha, bool(status)


def alias_exists(client: MlflowClient, model_name: str) -> bool:
    try:
        client.get_model_version_by_alias(model_name, ALIAS)
    except MlflowException:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and log the turnout model.")
    parser.add_argument("--data", default="data/modeling_table.parquet")
    parser.add_argument("--dvc-file", default="data/modeling_table.parquet.dvc")
    parser.add_argument("--dataset-version", required=True, help="v1, v2, ...")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument(
        "--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--register", action="store_true", help="register the model, no alias"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="initial release: register and set the alias champion",
    )
    parser.add_argument("--output", help="write a JSON summary of the candidate")
    args = parser.parse_args()

    client = MlflowClient()
    if args.bootstrap and alias_exists(client, args.model_name):
        raise SystemExit(
            f"'{ALIAS}' already exists for {args.model_name}: "
            "--bootstrap is only for the initial release."
        )

    table = pd.read_parquet(args.data)
    train, evaluation = split_train_eval(table)
    params = {"max_iter": args.max_iter, "learning_rate": args.learning_rate}

    model = build_model(**params).fit(train[FEATURES], train[TARGET])
    predictions = predict_turnout(model, evaluation)
    metrics = {
        "mae": float(mean_absolute_error(evaluation[TARGET], predictions)),
        "rmse": float(root_mean_squared_error(evaluation[TARGET], predictions)),
        "mae_persistence": persistence_mae(evaluation),
    }
    metrics["mae_gain_vs_persistence"] = metrics["mae_persistence"] - metrics["mae"]
    metrics.update(chronological_metrics(table, params))

    git_sha, git_dirty = git_state()
    dvc_md5 = read_dvc_md5(Path(args.dvc_file))
    register = args.register or args.bootstrap
    # Quality floor of the initial release: beat the persistence baseline.
    gate_passed = metrics["mae"] < metrics["mae_persistence"]
    if args.bootstrap and not gate_passed:
        register = False

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"train-{args.dataset_version}") as run:
        mlflow.log_params(
            {
                "git_sha": git_sha,
                "git_dirty": git_dirty,
                "dataset_version": args.dataset_version,
                "dvc_md5": dvc_md5,
                "features": ",".join(FEATURES),
                "holdout_share": f"1/{HOLDOUT_SHARE}",
                "n_train": len(train),
                "n_eval": len(evaluation),
                "model": "HistGradientBoostingRegressor",
                **params,
            }
        )
        mlflow.log_metrics(metrics)
        report = {
            "dataset_version": args.dataset_version,
            "dvc_md5": dvc_md5,
            "git_sha": git_sha,
            "metric": "MAE in turnout points, lower is better",
            "metrics": metrics,
        }
        mlflow.log_dict(report, "evaluation.json")
        mlflow.log_artifact(args.dvc_file, artifact_path="dataset")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=infer_signature(evaluation[FEATURES], predictions),
            input_example=evaluation[FEATURES].head(3),
            registered_model_name=args.model_name if register else None,
            # cloudpickle keeps the slim API image free of extra dependencies.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    version = None
    if register:
        version = getattr(model_info, "registered_model_version", None)
        if version is None:
            versions = client.search_model_versions(f"name='{args.model_name}'")
            version = max(int(v.version) for v in versions)
        for key, value in {
            "dataset_version": args.dataset_version,
            "dvc_md5": dvc_md5,
            "git_sha": git_sha,
            "mae": f"{metrics['mae']:.4f}",
            "mae_persistence": f"{metrics['mae_persistence']:.4f}",
        }.items():
            client.set_model_version_tag(args.model_name, version, key, value)
        if args.bootstrap:
            client.set_registered_model_alias(args.model_name, ALIAS, version)

    summary = {**report, "run_id": run.info.run_id, "model_version": version}
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
    if args.bootstrap and not gate_passed:
        raise SystemExit("Quality floor not met: the model was not registered.")


if __name__ == "__main__":
    main()
