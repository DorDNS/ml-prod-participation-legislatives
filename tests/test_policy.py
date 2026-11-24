import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from ml.policy import (
    DRIFT_SHIFT,
    KEEP,
    MIN_IMPROVEMENT,
    PROMOTE,
    RETRAIN_MAE_INCREASE,
    decide_promotion,
    drift_detected,
    main,
    passes_quality_floor,
    retrain_needed,
    validate_rollback_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ["1", "2", "3"]


# Quality floor


def test_quality_floor_is_strict():
    assert passes_quality_floor(3.0, 3.5) is True
    assert passes_quality_floor(3.5, 3.5) is False
    assert passes_quality_floor(4.0, 3.5) is False


def test_floor_not_met_keeps_champion_even_if_candidate_is_better():
    # Candidate beats the champion by 1 point but not the persistence baseline
    decision = decide_promotion(
        candidate_mae=4.0, persistence_mae=3.5, champion_mae=5.0
    )
    assert decision.action == KEEP
    assert "persistence" in decision.reason


# Promotion against a champion


def test_gain_exactly_min_improvement_is_promoted():
    # 3.05 - 3.00 is 0.04999... in floating point: rounding must handle it
    decision = decide_promotion(
        candidate_mae=3.00, persistence_mae=4.0, champion_mae=3.05
    )
    assert MIN_IMPROVEMENT == 0.05
    assert decision.action == PROMOTE
    assert decision.gain == pytest.approx(0.05)


def test_gain_below_min_improvement_is_kept():
    decision = decide_promotion(
        candidate_mae=3.00, persistence_mae=4.0, champion_mae=3.04
    )
    assert decision.action == KEEP
    assert decision.gain == pytest.approx(0.04)


def test_candidate_worse_than_champion_is_kept():
    decision = decide_promotion(
        candidate_mae=3.2, persistence_mae=4.0, champion_mae=3.0
    )
    assert decision.action == KEEP
    assert decision.gain < 0


# No champion yet


def test_no_champion_and_floor_met_is_promoted():
    decision = decide_promotion(candidate_mae=3.0, persistence_mae=4.0)
    assert decision.action == PROMOTE
    assert decision.gain is None


def test_no_champion_and_floor_not_met_is_kept():
    decision = decide_promotion(candidate_mae=4.5, persistence_mae=4.0)
    assert decision.action == KEEP


# Invalid inputs


@pytest.mark.parametrize(
    "candidate, persistence, champion",
    [
        (math.nan, 4.0, 3.0),
        (3.0, math.nan, 3.0),
        (3.0, 4.0, math.nan),
        (math.inf, 4.0, 3.0),
        (-1.0, 4.0, 3.0),
        (3.0, 4.0, -0.5),
    ],
)
def test_invalid_mae_raises_value_error(candidate, persistence, champion):
    with pytest.raises(ValueError):
        decide_promotion(candidate, persistence, champion)


# Retraining


def test_new_data_version_triggers_retraining():
    needed, reason = retrain_needed("v1", "v2", 3.0, 3.0)
    assert needed is True
    assert "v2" in reason


def test_mae_increase_at_threshold_triggers_retraining():
    assert RETRAIN_MAE_INCREASE == 1.0
    needed, _ = retrain_needed("v1", "v1", 3.0, 4.0)
    assert needed is True


def test_mae_increase_below_threshold_does_not_trigger_retraining():
    needed, _ = retrain_needed("v1", "v1", 3.0, 3.99)
    assert needed is False


# Drift


def test_drift_at_threshold_is_detected():
    assert DRIFT_SHIFT == 5.0
    assert drift_detected(50.0, 55.0) is True


def test_drift_below_threshold_is_not_detected():
    assert drift_detected(50.0, 54.9) is False


def test_negative_drift_is_detected():
    assert drift_detected(50.0, 44.0) is True


# Rollback


def test_rollback_to_missing_version_raises():
    with pytest.raises(ValueError, match="does not exist"):
        validate_rollback_target("9", current_version="3", existing_versions=VERSIONS)


def test_rollback_to_current_version_raises():
    with pytest.raises(ValueError, match="already"):
        validate_rollback_target("3", current_version="3", existing_versions=VERSIONS)


def test_rollback_to_valid_version_passes():
    validate_rollback_target("2", current_version="3", existing_versions=VERSIONS)


# Command line


def _write_candidate(path, metrics):
    path.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")


def test_cli_promotes_and_writes_decision(tmp_path):
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "decision.json"
    _write_candidate(
        candidate, {"mae": 3.0, "mae_persistence": 4.0, "champion_mae": 3.5}
    )

    exit_code = main(
        ["decide", "--candidate", str(candidate), "--output", str(output)]
    )

    assert exit_code == 0
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["action"] == PROMOTE
    assert decision["candidate_mae"] == 3.0
    assert decision["champion_mae"] == 3.5
    assert decision["gain"] == pytest.approx(0.5)
    assert decision["thresholds"]["min_improvement"] == MIN_IMPROVEMENT
    assert decision["reason"]


def test_cli_without_champion_mae(tmp_path):
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "decision.json"
    _write_candidate(candidate, {"mae": 3.0, "mae_persistence": 4.0})

    main(["decide", "--candidate", str(candidate), "--output", str(output)])

    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["action"] == PROMOTE
    assert decision["champion_mae"] is None


def test_cli_keep_exits_with_zero_via_python_m(tmp_path):
    # Runs the real command used by ct.yml: python -m ml.policy decide ...
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "decision.json"
    _write_candidate(
        candidate, {"mae": 3.0, "mae_persistence": 4.0, "champion_mae": 3.01}
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "ml.policy", "decide",
            "--candidate", str(candidate), "--output", str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["action"] == KEEP
