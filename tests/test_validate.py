import numpy as np
import pandas as pd
import pytest

from ml import validate as validate_module
from ml.train import load_table
from ml.validate import (
    DataValidationError,
    check_split,
    check_table,
    main,
    validate,
)


def make_table(stations: int = 200) -> pd.DataFrame:
    """A valid table: three elections, the same stations each time."""
    rng = np.random.default_rng(0)
    frames = []
    for year in (2012, 2017, 2022):
        frames.append(
            pd.DataFrame(
                {
                    "id_brut_miom": [f"st{i:04d}" for i in range(stations)],
                    "election_year": year,
                    "prev_turnout": rng.uniform(40, 80, stations),
                    "prev2_turnout": rng.uniform(40, 80, stations),
                    "registered": rng.integers(200, 3000, stations).astype("float64"),
                    "dept_code": rng.choice(["01", "13", "2A", "75"], stations),
                    "turnout": rng.uniform(40, 80, stations),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_valid_table_passes():
    validate(make_table())


def test_edge_values_are_accepted():
    table = make_table()
    table.loc[0, "turnout"] = 0.0
    table.loc[1, "turnout"] = 100.0
    table.loc[2, "registered"] = 1.0
    validate(table)


def test_missing_column_is_reported():
    problems = check_table(make_table().drop(columns=["registered", "dept_code"]))
    assert problems == ["missing columns: registered, dept_code"]


def test_empty_table_is_reported():
    assert check_table(make_table().iloc[0:0]) == ["the table is empty"]


@pytest.mark.parametrize(
    "column, bad_value, expected",
    [
        ("turnout", np.nan, "turnout: 1 missing values"),
        ("prev_turnout", np.nan, "prev_turnout: 1 missing values"),
        ("turnout", 100.5, "turnout: 1 values outside [0, 100]"),
        ("turnout", -0.1, "turnout: 1 values outside [0, 100]"),
        ("prev_turnout", 120.0, "prev_turnout: 1 values outside [0, 100]"),
        ("prev2_turnout", -5.0, "prev2_turnout: 1 values outside [0, 100]"),
        ("registered", 0.0, "registered: 1 values that are not positive"),
        ("registered", -10.0, "registered: 1 values that are not positive"),
        ("registered", "many", "registered: 1 values are not numbers"),
        ("election_year", 1800, "election_year: 1 values outside [1958, 2100]"),
        ("dept_code", "", "dept_code: 1 empty department codes"),
        ("dept_code", "  ", "dept_code: 1 empty department codes"),
        ("dept_code", None, "dept_code: 1 missing values"),
        ("id_brut_miom", None, "id_brut_miom: 1 missing values"),
    ],
)
def test_bad_value_is_reported(column, bad_value, expected):
    table = make_table()
    table[column] = table[column].astype(object)
    table.loc[5, column] = bad_value
    assert expected in check_table(table)


def test_duplicated_station_in_one_election_is_reported():
    table = make_table()
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    problems = check_table(table)
    assert "1 polling stations appear twice in the same election" in problems


def test_every_problem_is_listed():
    table = make_table()
    table.loc[0, "turnout"] = 150.0
    table.loc[1, "registered"] = 0.0
    with pytest.raises(DataValidationError) as error:
        validate(table)
    assert len(error.value.problems) == 2
    assert "turnout" in str(error.value) and "registered" in str(error.value)


def test_split_overlap_is_reported(monkeypatch):
    table = make_table()
    latest = table[table["election_year"] == 2022]

    def leaky_split(data):
        return latest.head(10), latest.head(5)

    monkeypatch.setattr(validate_module, "split_train_eval", leaky_split)
    with pytest.raises(DataValidationError, match="in both the training"):
        validate(table)


def test_evaluation_from_an_older_election_is_reported():
    table = make_table()
    old = table[table["election_year"] == 2012]
    train = table[table["election_year"] == 2022]
    problems = check_split(table, train, old.head(5))
    assert any("latest election (2022)" in problem for problem in problems)


def test_empty_evaluation_or_training_is_reported():
    table = make_table()
    assert check_split(table, table, table.iloc[0:0]) == [
        "the evaluation rows are empty"
    ]
    assert check_split(table, table.iloc[0:0], table) == [
        "the training rows are empty"
    ]


def test_real_split_is_separated():
    table = make_table()
    assert check_split(table, *validate_module.split_train_eval(table)) == []


def test_cli_accepts_a_valid_file(tmp_path, capsys):
    path = tmp_path / "table.parquet"
    make_table().to_parquet(path, index=False)
    assert main(["--data", str(path)]) == 0
    assert "Data valid" in capsys.readouterr().out


def test_cli_rejects_an_invalid_file(tmp_path, capsys):
    table = make_table()
    table.loc[0, "turnout"] = 150.0
    path = tmp_path / "table.parquet"
    table.to_parquet(path, index=False)
    assert main(["--data", str(path)]) == 1
    assert "outside [0, 100]" in capsys.readouterr().err


def test_training_stops_on_an_invalid_table(tmp_path):
    table = make_table()
    table.loc[0, "registered"] = 0.0
    path = tmp_path / "table.parquet"
    table.to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="registered"):
        load_table(str(path))


def test_training_loads_a_valid_table(tmp_path):
    path = tmp_path / "table.parquet"
    make_table().to_parquet(path, index=False)
    assert len(load_table(str(path))) == 600
