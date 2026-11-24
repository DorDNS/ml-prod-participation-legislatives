# Data

## Source

- French legislative elections, first round, results per polling station.
- Dataset: "Données des élections agrégées" on data.gouv.fr.
- License and usage terms: TODO (to be verified and recorded here).
- Scope: metropolitan France only (overseas codes starting with "Z" are excluded).

## Model contract (draft, to be confirmed after the first baseline)

- Registered model name: `polling-station-turnout`, alias `champion`.
- Task: supervised regression.
- Target: turnout in percent of registered voters (voters / registered * 100).
- Features: turnout of the previous election (%), turnout two elections ago (%),
  number of registered voters, department code.
- Primary metric: mean absolute error (MAE) in percentage points, lower is better.

## Dataset versions

TODO: for each version (v1, v2), record the Git tag, the DVC hash, what changed
between versions and the train/evaluation split design.
