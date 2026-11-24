"""Move the ``champion`` alias of the model registry: promotion and rollback.

These are the only two operations that change which model production serves.
They are used by the continuous training workflow and can be run by hand.

A version is "approved" once it has been the champion (it is tagged
``approved=true`` when it is promoted, and the champion it replaces is tagged
too). A rollback is only allowed to an approved version: a candidate that was
registered but rejected by the promotion rule can never become the champion
through a rollback.

Usage (MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD must be set):

    python -m ml.promote promote --version 2 --output promotion.json
    python -m ml.promote rollback --version 1 --output rollback.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.policy import validate_rollback_target

DEFAULT_MODEL_NAME = "polling-station-turnout"
ALIAS = "champion"
APPROVED_TAG = "approved"


def existing_versions(client: MlflowClient, model_name: str) -> set[str]:
    """Version numbers registered for the model, as strings."""
    versions = client.search_model_versions(f"name='{model_name}'")
    return {str(version.version) for version in versions}


def current_champion(client: MlflowClient, model_name: str) -> str | None:
    """Version that carries the champion alias, or None if there is none."""
    try:
        return str(client.get_model_version_by_alias(model_name, ALIAS).version)
    except MlflowException:
        return None


def promote(client: MlflowClient, model_name: str, version: str) -> dict:
    """Make ``version`` the champion and mark it, and the old champion, approved."""
    version = str(version)
    if version not in existing_versions(client, model_name):
        raise ValueError(f"Version {version!r} of {model_name!r} does not exist.")
    previous = current_champion(client, model_name)
    if previous == version:
        raise ValueError(f"Version {version!r} is already the champion.")
    if previous is not None:
        client.set_model_version_tag(model_name, previous, APPROVED_TAG, "true")
    client.set_model_version_tag(model_name, version, APPROVED_TAG, "true")
    client.set_registered_model_alias(model_name, ALIAS, version)
    return {"action": "promote", "previous_version": previous, "new_version": version}


def rollback(client: MlflowClient, model_name: str, version: str) -> dict:
    """Move the champion alias back to an approved version."""
    version = str(version)
    versions = existing_versions(client, model_name)
    previous = current_champion(client, model_name)
    if previous is None:
        raise ValueError(f"{model_name!r} has no champion: nothing to roll back.")
    validate_rollback_target(version, previous, versions)
    tags = client.get_model_version(model_name, version).tags or {}
    if tags.get(APPROVED_TAG) != "true":
        raise ValueError(
            f"Version {version!r} was never the champion (not approved): "
            "refusing to roll back to it."
        )
    client.set_registered_model_alias(model_name, ALIAS, version)
    return {"action": "rollback", "previous_version": previous, "new_version": version}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.promote")
    parser.add_argument("operation", choices=["promote", "rollback"])
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    )
    parser.add_argument("--output", type=Path, help="write the result as JSON")
    args = parser.parse_args(argv)

    client = MlflowClient()
    operation = promote if args.operation == "promote" else rollback
    try:
        result = operation(client, args.model_name, args.version)
    except ValueError as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 1
    print(
        f"{result['action']}: champion moved from version "
        f"{result['previous_version']} to {result['new_version']}"
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
