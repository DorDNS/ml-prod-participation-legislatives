# Experiments

All runs are logged in the shared MLflow server on DagsHub, experiment `turnout-regression`. Each run records the Git commit (`git_sha`), the dataset version and its DVC hash, the hyperparameters, the metrics, the model and an `evaluation.json` file. Data versions and the train/evaluation split are described in [README_data.md](README_data.md).

## Primary metric

The primary metric is the **mean absolute error (MAE) of the predicted turnout, in percentage points. Lower is better.**

Why this metric: the web application returns a turnout percentage for one polling station, and the MAE is the average gap between that prediction and the real value, in the same unit. A MAE of 3 means the prediction is off by 3 points on average. It is easy to explain and less sensitive to a few extreme stations than the RMSE, which is logged as a secondary metric.

Every model is compared with a **persistence baseline**: predict that a station repeats its previous turnout (`prev_turnout`). A model is only useful if it beats this baseline.

## Model and protocol

- Model: scikit-learn pipeline, ordinal encoding of the department, then `HistGradientBoostingRegressor` (`max_iter=300`, `learning_rate=0.05`, `l2_regularization=1.0`, `random_state=42`). Predictions are clipped to the range 0 to 100.
- Features (contract of the API): `prev_turnout`, `prev2_turnout`, `registered`, `dept_code`, `election_year`.
- Evaluation: one polling station out of five of the latest election of the dataset version is held out (election-night nowcasting). The candidate and the champion are always evaluated on the same rows.
- Complementary check: a strictly chronological evaluation (train on earlier elections only, test on the latest one), logged as `chrono_mae`.
- No hyperparameter search was run: the project focuses on the MLOps pipeline, so a single configuration is used.

## Quality gates

The gates are defined before any evaluation, in `ml/policy.py`, which is the single source of truth for the thresholds. The gain is the champion MAE minus the candidate MAE, so a positive gain means the candidate is better.

| Gate | Rule | Constant in `ml/policy.py` |
|---|---|---|
| Quality floor | The candidate MAE must be strictly lower than the persistence MAE, otherwise it is never promoted | `candidate_mae < persistence_mae` |
| Minimum improvement | The candidate is promoted only if its gain over the champion, on the same rows, is at least 0.05 points. Otherwise the champion is kept | `MIN_IMPROVEMENT = 0.05` |
| Retraining trigger | A new dataset version, or a champion MAE that increased by at least 1.0 point | `RETRAIN_MAE_INCREASE = 1.0` |
| Drift alert | Mean turnout shifted by at least 5.0 points. Logged, never blocking | `DRIFT_SHIFT = 5.0` |
| Rollback | Only to a version that was champion before (tag `approved=true`), and never to the current champion | `validate_rollback_target` |

The first two gates are applied by the `decide` job of the `ct.yml` workflow. The retraining trigger and the drift alert are implemented and unit tested, but the workflow does not use them as a gate: it retrains every time it is started manually.

## Results

| Run | Dataset | Train rows | Evaluation rows | MAE | RMSE | Persistence MAE | Gain vs persistence |
|---|---|---|---|---|---|---|---|
| `train-v1` (`0844de69`) | v1 | 170,575 | 12,247 | 3.2556 | 4.4126 | 3.8260 | 0.5704 |
| `train-v2` (`9a91fafe`) | v2 | 235,146 | 12,917 | 2.9369 | 3.9656 | 19.0007 | 16.0638 |

Strictly chronological check (train on earlier elections, test on the latest one):

| Dataset | Model MAE | Persistence MAE |
|---|---|---|
| v1 (test on 2022) | 5.3603 | 3.8001 |
| v2 (test on 2024) | 20.8730 | 18.9590 |

## Promotion decisions (workflow `ct.yml`)

| Run | Operation | Candidate MAE | Champion MAE | Gain | Decision | Duration |
|---|---|---|---|---|---|---|
| Continuous training #1 | pipeline, dataset v2 | 2.9369 | 20.9722 (version 1) | 18.0353 | **promote** (gain above 0.05, quality floor passed): production serves version 2 after redeployment | 3m30s |
| Continuous training #2 | pipeline, dataset v2 | 2.9369 | 2.9369 (version 2) | 0.0000 | **keep**: gain below 0.05. The candidate (`2aba7123`, version 3) is registered but not promoted | 1m47s |
| Continuous training #3 | rollback to version 1 | | | | champion back to version 1, checked with `/model-info` (`run_id` `0844de69`) | 1m31s |

## Evidence for the production choice

- On the held-out stations of 2024, version 2 reaches a MAE of 2.94 points. Version 1, which never saw the 2024 data, reaches 20.97 points on the same rows. The gain of 18.04 points is far above the minimum improvement of 0.05, and the quality floor is met (2.94 against a persistence MAE of 19.00).
- Version 2 is therefore the right champion for election-night use, when part of the stations of the election are already known. Version 1 stays available as an approved rollback target.
- The rollback of Continuous training #3 shows that the previous version can be restored and verified in production in about 1.5 minutes.

## Limits

- **The chronological check is less favorable than the persistence baseline** for both versions (5.36 against 3.80 for v1, 20.87 against 18.96 for v2). The model is good at predicting the remaining stations of an election whose first results are known, but it does not beat "repeat the previous turnout" when predicting a whole new election from past elections only. The model must not be presented as a forecast of a future election.
- The very large gain of version 2 over version 1 mostly reflects a change of turnout level between 2022 and 2024: the persistence MAE jumps from 3.8 to 19.0 points. Version 1 had no way to know it. This shows why retraining on new data matters, but it is not evidence of a better modelling method.
- A single model configuration and a single seed were evaluated.
- Stations with extreme turnout values (0 % or 100 %) are kept in the data.
