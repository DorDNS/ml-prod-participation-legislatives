import pandas as pd
import pytest

from ml.dataset import (
    build_modeling_table,
    load_legislative_turnout,
    split_train_eval,
)


def make_turnout(values: dict[tuple[str, int], float]) -> pd.DataFrame:
    rows = [
        {
            "id_brut_miom": station,
            "election_year": year,
            "dept_code": "75",
            "registered": 1000.0,
            "turnout": turnout,
        }
        for (station, year), turnout in values.items()
    ]
    return pd.DataFrame(rows)


def test_load_keeps_only_metropolitan_legislative_first_rounds(tmp_path):
    raw = pd.DataFrame(
        {
            "id_election": [
                "2022_legi_t1",
                "2022_legi_t1",
                "2022_legi_t2",
                "2022_legi_t1",
                "2022_pres_t1",
            ],
            "id_brut_miom": [
                "01001_0001",
                "ZA001_0001",
                "01001_0001",
                "01001_0002",
                "01001_0001",
            ],
            "code_departement": ["01", "ZA", "01", "01", "01"],
            "inscrits": [200, 200, 200, 0, 200],
            "votants": [100, 100, 100, 0, 100],
        }
    )
    path = tmp_path / "raw.parquet"
    raw.to_parquet(path)
    result = load_legislative_turnout(str(path))
    assert list(result["id_brut_miom"]) == ["01001_0001"]
    assert result["turnout"].iloc[0] == 50.0


def test_build_adds_the_two_previous_turnouts():
    values = {
        ("a", 2012): 50.0,
        ("a", 2017): 60.0,
        ("a", 2022): 70.0,
        ("b", 2012): 40.0,
        ("b", 2017): 45.0,
        ("b", 2022): 55.0,
    }
    table = build_modeling_table(make_turnout(values), 2022)
    row = table[table["id_brut_miom"] == "a"].iloc[0]
    assert len(table) == 2
    assert row["prev_turnout"] == 60.0
    assert row["prev2_turnout"] == 50.0
    assert row["turnout"] == 70.0


def test_station_without_history_is_dropped():
    values = {
        ("a", 2012): 50.0,
        ("a", 2017): 60.0,
        ("a", 2022): 70.0,
        ("c", 2017): 45.0,
        ("c", 2022): 55.0,
    }
    table = build_modeling_table(make_turnout(values), 2022)
    assert set(table["id_brut_miom"]) == {"a"}


def test_max_year_excludes_later_elections():
    values = {
        (station, year): 50.0
        for station in ("a", "b")
        for year in (2007, 2012, 2017, 2022, 2024)
    }
    table = build_modeling_table(make_turnout(values), 2022)
    assert table["election_year"].max() == 2022


def test_at_least_three_elections_are_required():
    values = {("a", 2017): 60.0, ("a", 2022): 70.0}
    with pytest.raises(ValueError):
        build_modeling_table(make_turnout(values), 2022)


def test_split_holds_out_part_of_the_latest_election_only():
    values = {
        (f"s{i}", year): 50.0 + (i % 10)
        for i in range(300)
        for year in (2007, 2012, 2017, 2022)
    }
    table = build_modeling_table(make_turnout(values), 2022)
    train, evaluation = split_train_eval(table)
    again = split_train_eval(table)[1]
    latest = table[table["election_year"] == 2022]
    assert len(train) + len(evaluation) == len(table)
    assert set(evaluation["election_year"]) == {2022}
    assert evaluation.index.equals(again.index)
    assert 0.1 < len(evaluation) / len(latest) < 0.3
