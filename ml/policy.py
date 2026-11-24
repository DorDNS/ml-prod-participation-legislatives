"""Decision rules for the continuous training (CT) workflow.

This module decides whether a newly trained candidate model is promoted to
``champion`` or whether the current champion is kept. It also decides when
retraining is needed, raises drift alerts and validates rollback targets.

Metric: mean absolute error (MAE) of the predicted turnout, in percentage
points. LOWER IS BETTER: a model with a MAE of 3.0 is better than a model
with a MAE of 3.5.

All comparisons between models are made on the same rows (same evaluation
set). The functions are pure Python: no MLflow import, no network and no
file access, so they are fast and easy to test. The command line interface
at the bottom of the file is the only part that reads and writes files.

To change a threshold, change the constant below: it is the single source
of truth, referenced by ``ml/README_experiments.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Minimum MAE gain (in points) the candidate must have over the champion,
# measured on the same rows, to be promoted.
MIN_IMPROVEMENT = 0.05

# Increase of the champion's MAE (in points) that triggers a retraining.
RETRAIN_MAE_INCREASE = 1.0

# Shift of the mean turnout (in points) between the reference data and the
# new data that raises a drift alert. The alert is logged, it never blocks.
DRIFT_SHIFT = 5.0

# Number of decimals used when comparing a gain to MIN_IMPROVEMENT, to avoid
# floating point errors at the boundary (e.g. 3.05 - 3.00 = 0.04999...).
GAIN_DECIMALS = 6

PROMOTE = "promote"
KEEP = "keep"


@dataclass(frozen=True)
class Decision:
    """Outcome of the promotion rule.

    action: ``"promote"`` or ``"keep"``.
    reason: human readable explanation, logged by the CT workflow.
    gain: champion MAE minus candidate MAE (positive means the candidate is
        better), or ``None`` when there is no champion.
    """

    action: str
    reason: str
    gain: float | None


def _check_mae(name: str, value: float) -> None:
    """Raise ValueError if a MAE is NaN, infinite or negative."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")


def passes_quality_floor(candidate_mae: float, persistence_mae: float) -> bool:
    """Return True if the candidate strictly beats the persistence baseline.

    The persistence baseline predicts the same turnout as the previous
    election, computed on the same rows as the candidate. Lower MAE is
    better, so the candidate must have a strictly lower MAE.
    """
    _check_mae("candidate_mae", candidate_mae)
    _check_mae("persistence_mae", persistence_mae)
    return candidate_mae < persistence_mae


def decide_promotion(
    candidate_mae: float,
    persistence_mae: float,
    champion_mae: float | None = None,
) -> Decision:
    """Decide whether the candidate is promoted to champion.

    Rules, in order:
    1. If the candidate does not beat the persistence baseline, keep the
       champion, even if the candidate is better than the champion.
    2. If there is no champion yet, promote the candidate.
    3. Otherwise, promote only if the MAE gain over the champion is at least
       MIN_IMPROVEMENT points.

    Raises ValueError if a MAE is NaN, infinite or negative.
    """
    _check_mae("candidate_mae", candidate_mae)
    _check_mae("persistence_mae", persistence_mae)
    if champion_mae is not None:
        _check_mae("champion_mae", champion_mae)

    gain = None if champion_mae is None else champion_mae - candidate_mae

    if not passes_quality_floor(candidate_mae, persistence_mae):
        return Decision(
            action=KEEP,
            reason=(
                f"Candidate MAE {candidate_mae:.4f} does not beat the "
                f"persistence baseline MAE {persistence_mae:.4f} "
                "(quality floor not met)."
            ),
            gain=gain,
        )

    if champion_mae is None:
        return Decision(
            action=PROMOTE,
            reason=(
                "No current champion and the candidate beats the "
                "persistence baseline."
            ),
            gain=None,
        )

    if round(gain, GAIN_DECIMALS) >= MIN_IMPROVEMENT:
        return Decision(
            action=PROMOTE,
            reason=(
                f"Candidate improves the champion MAE by {gain:.4f} points "
                f"(minimum {MIN_IMPROVEMENT})."
            ),
            gain=gain,
        )

    return Decision(
        action=KEEP,
        reason=(
            f"Candidate gain over the champion is {gain:.4f} points, "
            f"below the minimum of {MIN_IMPROVEMENT}."
        ),
        gain=gain,
    )


def retrain_needed(
    champion_dataset_version: str,
    new_dataset_version: str,
    champion_mae_reference: float,
    champion_mae_now: float,
) -> tuple[bool, str]:
    """Return (needed, reason) telling whether a retraining is needed.

    Retraining is needed if a new data version is available, or if the
    champion's MAE has increased by at least RETRAIN_MAE_INCREASE points
    compared to its reference MAE (higher MAE means worse performance).
    """
    if new_dataset_version != champion_dataset_version:
        return True, (
            f"New data version {new_dataset_version!r} differs from the "
            f"champion's data version {champion_dataset_version!r}."
        )

    increase = champion_mae_now - champion_mae_reference
    if round(increase, GAIN_DECIMALS) >= RETRAIN_MAE_INCREASE:
        return True, (
            f"Champion MAE increased by {increase:.4f} points "
            f"(threshold {RETRAIN_MAE_INCREASE})."
        )

    return False, (
        f"Same data version and champion MAE increase of {increase:.4f} "
        f"points is below the threshold of {RETRAIN_MAE_INCREASE}."
    )


def drift_detected(reference_mean: float, new_mean: float) -> bool:
    """Return True if the mean turnout shifted by at least DRIFT_SHIFT points.

    The shift is measured in both directions. This is an alert to log, it
    does not block the workflow.
    """
    shift = abs(new_mean - reference_mean)
    return round(shift, GAIN_DECIMALS) >= DRIFT_SHIFT


def validate_rollback_target(
    target_version: str,
    current_version: str,
    existing_versions: list[str] | set[str],
) -> None:
    """Raise ValueError if the rollback target is not a valid choice.

    The target must exist in the registry and must not already be the
    current champion version.
    """
    if target_version not in existing_versions:
        raise ValueError(
            f"Rollback target version {target_version!r} does not exist."
        )
    if target_version == current_version:
        raise ValueError(
            f"Rollback target version {target_version!r} is already the "
            "current champion version."
        )


def _decide_command(candidate_path: Path, output_path: Path) -> dict:
    """Read candidate.json, apply the promotion rule, write decision.json."""
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    metrics = candidate["metrics"]
    candidate_mae = float(metrics["mae"])
    persistence_mae = float(metrics["mae_persistence"])
    champion_mae = metrics.get("champion_mae")
    if champion_mae is not None:
        champion_mae = float(champion_mae)

    decision = decide_promotion(candidate_mae, persistence_mae, champion_mae)

    result = {
        **asdict(decision),
        "candidate_mae": candidate_mae,
        "champion_mae": champion_mae,
        "persistence_mae": persistence_mae,
        "thresholds": {
            "min_improvement": MIN_IMPROVEMENT,
            "quality_floor": "candidate_mae < persistence_mae",
        },
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    """Command line entry point, used by the CT workflow."""
    parser = argparse.ArgumentParser(prog="python -m ml.policy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    decide = subparsers.add_parser(
        "decide", help="Decide whether the candidate is promoted."
    )
    decide.add_argument("--candidate", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "decide":
        result = _decide_command(args.candidate, args.output)
        print(f"Decision: {result['action']} - {result['reason']}")
    # Exit code 0 for both "promote" and "keep": the workflow reads "action".
    return 0


if __name__ == "__main__":
    sys.exit(main())
