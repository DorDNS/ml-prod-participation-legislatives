# Data

## Source

- Dataset: "Données des élections agrégées" on data.gouv.fr, results per polling station.  The dataset is built from the result files published by the French Ministry of the Interior.
Its structure changed in January 2026, so the file format may differ in older copies.
- License and usage terms: [Licence Ouverte / Open Licence version 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence) (Etalab), as stated on the dataset page.
- Scope: legislative elections, first round only, metropolitan France only (overseas codes starting with "Z" are excluded), only polling stations with registered voters > 0.
- Raw file: `data/general_results.parquet`. It is not versioned in Git because it is too large.
- Download date: 23 November 2026.

## Task and target

- Task: supervised regression.
- Target: turnout in percent of registered voters (voters / registered * 100).
- The 5 model inputs (the API contract, in this order):
  `prev_turnout`, `prev2_turnout`, `registered`, `dept_code`, `election_year`.
- `prev_turnout` and `prev2_turnout` are the turnout of the same polling station at the two previous legislative elections. A polling station that does not have both is excluded.
- No column coming from the target election (voters, expressed votes, blank votes...) is used as a feature: this would be a data leak.
- Primary metric: described in `ml/README_experiments.md`.

## Dataset versions

| | v1 | v2 |
|---|---|---|
| Content | elections up to 2022, targets 2012, 2017, 2022 | v1 + 2024 |
| Rows | 182,822 | 248,063 |
| Git tag | `data-v1` (commit `25f1671`) | `data-v2` (commit `381e1af`) |
| DVC hash (md5) | `5885fb1158f684e5f163593e3344ab71` | `e90d5d5a15985027ed13b5819044097d` |
| Size | 3,890,923 bytes | 5,110,507 bytes |

- Change from v1 to v2: the 2024 election is added (+65,241 rows). It is also the most recent election, so it is the one used for evaluation in v2.
- Storage: DVC, with an Azure Blob remote (container `dvc`, path `ml-prod-participation-legislatives`). The pointer file is `data/modeling_table.parquet.dvc`. No token or SAS is stored in this repository.
- To get a given version back:

```
git checkout data-v1 -- data/modeling_table.parquet.dvc
dvc pull
```

(same with `data-v2`). This is what the continuous training workflow does.
- To check the hashes: `git show data-v1:data/modeling_table.parquet.dvc` and `git show data-v2:data/modeling_table.parquet.dvc`.

## Train / evaluation split

- Evaluation: 1 polling station out of 5 of the most recent election of the version (bucket 0 of `crc32(id_brut_miom)` modulo 5). Everything else, including the other polling stations of the same election, is used for training.
- Why: this matches the "election night" use case, where the results of some polling stations of the election are already known and the others must be predicted. The hash-based split is the same on every machine and does not depend on a random draw.
- The candidate and the champion are always compared on the same rows.
- Limit: a strictly chronological evaluation (train without the latest election, test on
it) is also computed and logged in MLflow (`chrono_mae`). It is less favorable than the
persistence baseline. See `ml/README_experiments.md` for the numbers.

## Data quality

- Extreme values (turnout of 0% or 100%) are kept and not corrected.