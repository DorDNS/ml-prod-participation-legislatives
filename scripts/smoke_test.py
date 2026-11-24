"""Smoke test for a running instance of the turnout API.

It checks that the service becomes ready, answers correctly and serves the
expected model version. It runs on a developer machine or in a GitHub runner,
never inside the Docker image.

Examples:
    python scripts/smoke_test.py --url http://localhost:8000
    python scripts/smoke_test.py --url https://my-app.onrender.com --expect-champion
    python scripts/smoke_test.py --url http://localhost:8000 --expect-version 3

Exit codes:
    0  every check passed
    1  a check failed (or the service was not ready in time)
    2  usage or configuration error
"""

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

# A sleeping Render service can keep a request pending for about a minute.
REQUEST_TIMEOUT = 70

VALID_PAYLOAD = {
    "prev_turnout": 60.0,
    "prev2_turnout": 62.0,
    "registered": 1200.0,
    "dept_code": "75",
    "election_year": 2024,
}
INVALID_PAYLOAD = {**VALID_PAYLOAD, "prev_turnout": 120.0}
MODEL_INFO_KEYS = ("model_name", "alias", "version", "run_id")
MLFLOW_VARIABLES = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


class CheckFailed(Exception):
    """A verification did not pass."""


class ConfigError(Exception):
    """The script is not configured correctly (for example missing variables)."""


def request_json(method, url, payload=None, timeout=REQUEST_TIMEOUT):
    """Send a request and return (status code, parsed JSON body or None)."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:  # 4xx and 5xx are answers, not crashes
        status, raw = error.code, error.read()
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    return status, body


def wait_until_ready(base_url, timeout, interval):
    """Poll GET /ready until it answers 200 {"ready": true}, or fail on timeout."""
    deadline = time.monotonic() + timeout
    last_problem = "no answer"
    while True:
        remaining = deadline - time.monotonic()
        try:
            status, body = request_json(
                "GET",
                f"{base_url}/ready",
                timeout=max(1, min(REQUEST_TIMEOUT, remaining)),
            )
            if status == 200 and isinstance(body, dict) and body.get("ready") is True:
                return "GET /ready answers 200 {\"ready\": true}"
            last_problem = f"status {status}"
        except (OSError, http.client.HTTPException) as error:
            # Connection refused, timeout... the service is probably still starting.
            last_problem = type(error).__name__
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CheckFailed(
                f"GET /ready was not ready after {timeout:g}s (last problem: {last_problem})"
            )
        time.sleep(min(interval, remaining))


def check_model_info(base_url):
    """Check GET /model-info and return (served version, message)."""
    status, body = request_json("GET", f"{base_url}/model-info")
    if status != 200:
        raise CheckFailed(f"GET /model-info: expected status 200, got {status}")
    if not isinstance(body, dict):
        raise CheckFailed("GET /model-info: the body is not a JSON object")
    missing = [key for key in MODEL_INFO_KEYS if key not in body]
    if missing:
        raise CheckFailed(f"GET /model-info: missing keys {missing}")
    version = str(body["version"])
    return version, f"GET /model-info has {', '.join(MODEL_INFO_KEYS)} (version {version})"


def check_valid_prediction(base_url, served_version):
    """A valid request must return a turnout in [0, 100] and the served version."""
    status, body = request_json("POST", f"{base_url}/predict", VALID_PAYLOAD)
    if status != 200:
        raise CheckFailed(f"POST /predict: expected status 200, got {status}")
    if not isinstance(body, dict):
        raise CheckFailed("POST /predict: the body is not a JSON object")
    turnout = body.get("turnout_pct")
    if isinstance(turnout, bool) or not isinstance(turnout, int | float):
        raise CheckFailed(f"POST /predict: turnout_pct is not a number ({turnout!r})")
    if not 0 <= turnout <= 100:
        raise CheckFailed(f"POST /predict: turnout_pct {turnout} is outside [0, 100]")
    predicted_version = body.get("model_version")
    if not isinstance(predicted_version, str):
        raise CheckFailed("POST /predict: model_version is missing or not a string")
    if predicted_version != served_version:
        raise CheckFailed(
            f"POST /predict: model_version is {predicted_version}, "
            f"but /model-info says {served_version}"
        )
    return f"POST /predict returns turnout_pct {turnout} from model version {predicted_version}"


def check_invalid_prediction(base_url):
    """An out-of-range input must be rejected with 422."""
    status, _ = request_json("POST", f"{base_url}/predict", INVALID_PAYLOAD)
    if status != 422:
        raise CheckFailed(
            f"POST /predict with prev_turnout=120: expected status 422, got {status}"
        )
    return "POST /predict rejects prev_turnout=120 with 422"


def get_champion_version() -> str:
    """Read the version carried by the champion alias in the MLflow registry."""
    missing = [name for name in MLFLOW_VARIABLES if not os.environ.get(name)]
    if missing:
        raise ConfigError("missing environment variables: " + ", ".join(missing))
    # Imported here so the script works without MLflow when it is not needed.
    from mlflow import MlflowClient

    name = os.environ.get("MODEL_NAME", "polling-station-turnout")
    alias = os.environ.get("MODEL_ALIAS", "champion")
    return str(MlflowClient().get_model_version_by_alias(name, alias).version)


def check_expected_version(served_version, expected_version, source):
    """The served version must be the one we expect."""
    if served_version != expected_version:
        raise CheckFailed(
            f"served model version is {served_version}, expected {expected_version} ({source})"
        )
    return f"served model version {served_version} matches {source}"


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke test of a running turnout API.")
    parser.add_argument("--url", required=True, help="base URL, e.g. http://localhost:8000")
    parser.add_argument(
        "--timeout", type=float, default=300, help="seconds to wait for /ready"
    )
    parser.add_argument(
        "--interval", type=float, default=5, help="seconds between two /ready polls"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--expect-champion",
        action="store_true",
        help="the served version must be the one behind the champion alias",
    )
    group.add_argument(
        "--expect-version",
        metavar="N",
        help="the served version must be exactly N",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all checks and return the exit code."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exit_request:  # argparse exits with 2 on a usage error
        return int(exit_request.code or 0)

    base_url = args.url.rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        print("ERROR --url must start with http:// or https://", file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.interval < 0:
        print("ERROR --timeout must be > 0 and --interval must be >= 0", file=sys.stderr)
        return 2

    served_version = "unknown"
    try:
        print("OK  " + wait_until_ready(base_url, args.timeout, args.interval))
        served_version, message = check_model_info(base_url)
        print("OK  " + message)
        print("OK  " + check_valid_prediction(base_url, served_version))
        print("OK  " + check_invalid_prediction(base_url))
        if args.expect_version is not None:
            expected = args.expect_version.strip()
            message = check_expected_version(
                served_version, expected, f"--expect-version {expected}"
            )
            print("OK  " + message)
        elif args.expect_champion:
            try:
                champion = get_champion_version()
            except ConfigError:
                raise
            except Exception as error:  # registry unreachable, alias not found...
                raise CheckFailed(
                    f"cannot read the champion version ({type(error).__name__})"
                ) from error
            message = check_expected_version(
                served_version, champion, "the champion alias"
            )
            print("OK  " + message)
    except ConfigError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    except CheckFailed as error:
        print("FAIL " + str(error))
        print(f"Smoke test FAILED for {base_url}")
        return 1
    except (OSError, http.client.HTTPException) as error:
        print(f"FAIL cannot reach the service ({type(error).__name__})")
        print(f"Smoke test FAILED for {base_url}")
        return 1

    print(f"Smoke test PASSED: {base_url} serves model version {served_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())