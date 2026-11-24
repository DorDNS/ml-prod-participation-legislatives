"""Tests for scripts/smoke_test.py, against a fake local API (no real network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import smoke_test


class FakeApi:
    """A tiny HTTP server that imitates the turnout API."""

    def __init__(
        self,
        ready_after=0,
        never_ready=False,
        version="3",
        predict_version=None,
        turnout=55.0,
        invalid_status=422,
    ):
        self.ready_after = ready_after  # /ready answers 503 this many times first
        self.never_ready = never_ready
        self.version = version
        self.predict_version = predict_version or version
        self.turnout = turnout
        self.invalid_status = invalid_status
        self.ready_calls = 0
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # keep the test output quiet

            def send_json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/ready":
                    api.ready_calls += 1
                    if api.never_ready or api.ready_calls <= api.ready_after:
                        return self.send_json(503, {"ready": False})
                    return self.send_json(200, {"ready": True})
                if self.path == "/model-info":
                    return self.send_json(
                        200,
                        {
                            "model_name": "polling-station-turnout",
                            "alias": "champion",
                            "version": api.version,
                            "run_id": "run-abc",
                        },
                    )
                self.send_json(404, {"detail": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                if self.path != "/predict":
                    return self.send_json(404, {"detail": "not found"})
                if payload["prev_turnout"] > 100:
                    return self.send_json(api.invalid_status, {"detail": "invalid"})
                return self.send_json(
                    200,
                    {"turnout_pct": api.turnout, "model_version": api.predict_version},
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def start_api():
    """Start fake APIs and stop them all at the end of the test."""
    started = []

    def start(**options):
        api = FakeApi(**options)
        started.append(api)
        return api

    yield start
    for api in started:
        api.close()


def run(api, *extra):
    """Run the smoke test against a fake API, with short delays."""
    return smoke_test.main(
        ["--url", api.url, "--timeout", "5", "--interval", "0.1", *extra]
    )


def test_everything_is_correct(start_api, capsys):
    api = start_api()
    assert run(api) == 0
    output = capsys.readouterr().out
    assert "FAIL" not in output
    assert "PASSED" in output
    assert "version 3" in output


def test_cold_start_503_then_ready(start_api):
    api = start_api(ready_after=3)
    assert run(api) == 0
    assert api.ready_calls == 4


def test_never_ready_times_out(start_api):
    api = start_api(never_ready=True)
    assert run(api, "--timeout", "1") == 1


def test_unreachable_service_fails(start_api):
    api = start_api()
    api.close()  # nothing listens on this port any more
    assert run(api, "--timeout", "1") == 1


def test_predict_version_differs_from_model_info(start_api):
    api = start_api(version="3", predict_version="2")
    assert run(api) == 1


@pytest.mark.parametrize("turnout", [120.0, -1.0, "high"])
def test_turnout_out_of_range_or_not_a_number(start_api, turnout):
    api = start_api(turnout=turnout)
    assert run(api) == 1


def test_invalid_input_accepted_instead_of_422(start_api):
    api = start_api(invalid_status=200)
    assert run(api) == 1


def test_expect_version_that_does_not_match(start_api):
    api = start_api(version="3")
    assert run(api, "--expect-version", "4") == 1


def test_expect_version_that_matches(start_api):
    api = start_api(version="3")
    assert run(api, "--expect-version", "3") == 0


def test_expect_champion_that_matches(start_api, monkeypatch):
    monkeypatch.setattr(smoke_test, "get_champion_version", lambda: "3")
    api = start_api(version="3")
    assert run(api, "--expect-champion") == 0


def test_expect_champion_that_does_not_match(start_api, monkeypatch):
    monkeypatch.setattr(smoke_test, "get_champion_version", lambda: "5")
    api = start_api(version="3")
    assert run(api, "--expect-champion") == 1


def test_expect_champion_without_mlflow_variables(start_api, monkeypatch):
    for name in smoke_test.MLFLOW_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    api = start_api()
    assert run(api, "--expect-champion") == 2


def test_credentials_are_never_printed(start_api, monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "s3cret-token")

    def failing_lookup():
        raise RuntimeError("login failed for s3cret-token")

    monkeypatch.setattr(smoke_test, "get_champion_version", failing_lookup)
    api = start_api()
    assert run(api, "--expect-champion") == 1
    captured = capsys.readouterr()
    assert "s3cret-token" not in captured.out + captured.err


def test_url_without_scheme_is_a_usage_error():
    assert smoke_test.main(["--url", "localhost:8000"]) == 2


def test_both_expect_options_are_rejected():
    argv = ["--url", "http://localhost:8000", "--expect-champion", "--expect-version", "3"]
    assert smoke_test.main(argv) == 2