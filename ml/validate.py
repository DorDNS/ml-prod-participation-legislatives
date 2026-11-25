"""Check a modelling table before training: data contract and train/eval split.

A failure stops the run before anything is trained, registered or promoted.
Every problem found is listed, not only the first one.

Checks:
- the required columns exist and have no missing value;
- turnouts (target and the two previous ones) are percentages in [0, 100];
- the number of registered voters is positive;
- the department code and the election year are valid;
- a polling station appears at most once per election;
- the evaluation rows are the latest election only, and none of them is in the
  training rows.

Usage:
    python -m ml.validate --data data/modeling_table.parquet
Exit code: 0 when the table is valid, 1 otherwise.
"""

import argparse
import sys

import pandas as pd

from ml.dataset import FEATURES, TARGET, split_train_eval

ID_COLUMN = "id_brut_miom"
YEAR_COLUMN = "election_year"
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURES, TARGET]
PERCENT_COLUMNS = ["prev_turnout", "prev2_turnout", TARGET]
NUMERIC_COLUMNS = [*PERCENT_COLUMNS, "registered", YEAR_COLUMN]
MIN_YEAR, MAX_YEAR = 1958, 2100


class DataValidationError(ValueError):
    """The table does not respect the data contract."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Data validation failed:\n- " + "\n- ".join(self.problems))


def check_table(table: pd.DataFrame) -> list[str]:
    """Problems of the table itself (contract, values, labels)."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in table.columns]
    if missing_columns:
        return [f"missing columns: {', '.join(missing_columns)}"]
    if table.empty:
        return ["the table is empty"]

    problems = []
    for name in REQUIRED_COLUMNS:
        count = int(table[name].isna().sum())
        if count:
            problems.append(f"{name}: {count} missing values")

    numeric = {}
    for name in NUMERIC_COLUMNS:
        values = pd.to_numeric(table[name], errors="coerce")
        not_numeric = int((values.isna() & table[name].notna()).sum())
        if not_numeric:
            problems.append(f"{name}: {not_numeric} values are not numbers")
        numeric[name] = values

    for name in PERCENT_COLUMNS:
        values = numeric[name]
        count = int(((values < 0) | (values > 100)).sum())
        if count:
            problems.append(f"{name}: {count} values outside [0, 100]")
    count = int((numeric["registered"] <= 0).sum())
    if count:
        problems.append(f"registered: {count} values that are not positive")
    years = numeric[YEAR_COLUMN]
    count = int(((years < MIN_YEAR) | (years > MAX_YEAR)).sum())
    if count:
        problems.append(f"{YEAR_COLUMN}: {count} values outside [{MIN_YEAR}, {MAX_YEAR}]")
    departments = table["dept_code"].dropna().astype(str).str.strip()
    count = int((departments == "").sum())
    if count:
        problems.append(f"dept_code: {count} empty department codes")
    count = int(table.duplicated([ID_COLUMN, YEAR_COLUMN]).sum())
    if count:
        problems.append(f"{count} polling stations appear twice in the same election")
    return problems


def check_split(
    table: pd.DataFrame, train: pd.DataFrame, evaluation: pd.DataFrame
) -> list[str]:
    """Problems of the train/evaluation separation."""
    if train.empty:
        return ["the training rows are empty"]
    if evaluation.empty:
        return ["the evaluation rows are empty"]
    problems = []
    latest = table[YEAR_COLUMN].max()
    if set(evaluation[YEAR_COLUMN]) != {latest}:
        problems.append(f"the evaluation rows must all be from the latest election ({latest})")
    shared = train[[ID_COLUMN, YEAR_COLUMN]].merge(
        evaluation[[ID_COLUMN, YEAR_COLUMN]], how="inner"
    )
    if not shared.empty:
        problems.append(
            f"{len(shared)} polling station rows are in both the training and "
            "the evaluation rows"
        )
    return problems


def validate(table: pd.DataFrame) -> None:
    """Raise DataValidationError listing every problem of the table and its split."""
    problems = check_table(table)
    if not problems:
        train, evaluation = split_train_eval(table)
        problems = check_split(table, train, evaluation)
    if problems:
        raise DataValidationError(problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a modelling table.")
    parser.add_argument("--data", default="data/modeling_table.parquet")
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.data)
    try:
        validate(table)
    except DataValidationError as error:
        print(error, file=sys.stderr)
        return 1
    train, evaluation = split_train_eval(table)
    years = ", ".join(str(year) for year in sorted(table[YEAR_COLUMN].unique()))
    print(
        f"Data valid: {len(table)} rows, elections {years}; "
        f"{len(train)} training rows, {len(evaluation)} evaluation rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
