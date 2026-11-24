"""Build the modelling table from the aggregated election results.

One row per polling station and per legislative first round (metropolitan
France only), with the turnout of the two previous legislative first rounds
as features. A dataset version is defined by the last election it contains.

Usage:
    python -m ml.dataset --max-year 2022 --out data/modeling_table.parquet
"""

import argparse
import zlib

import pandas as pd

RAW_COLUMNS = [
    "id_election",
    "id_brut_miom",
    "code_departement",
    "inscrits",
    "votants",
]
FEATURES = [
    "prev_turnout",
    "prev2_turnout",
    "registered",
    "dept_code",
    "election_year",
]
TARGET = "turnout"
HOLDOUT_SHARE = 5  # one polling station out of five is kept for evaluation


def load_legislative_turnout(raw_path: str) -> pd.DataFrame:
    """Turnout (%) of every metropolitan polling station, legislative first rounds."""
    raw = pd.read_parquet(raw_path, columns=RAW_COLUMNS)
    is_legislative = raw["id_election"].str.endswith("_legi_t1")
    is_metropolitan = ~raw["code_departement"].str.startswith("Z", na=False)
    data = raw[is_legislative & is_metropolitan & (raw["inscrits"] > 0)].copy()
    return pd.DataFrame(
        {
            "id_brut_miom": data["id_brut_miom"],
            "election_year": data["id_election"].str[:4].astype(int),
            "dept_code": data["code_departement"],
            "registered": data["inscrits"].astype("float64"),
            "turnout": data["votants"] / data["inscrits"] * 100,
        }
    )


def build_modeling_table(turnout: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """One row per station and election up to max_year, with two previous turnouts.

    Stations without a turnout at both previous elections are dropped.
    """
    turnout = turnout[turnout["election_year"] <= max_year]
    years = sorted(turnout["election_year"].unique())
    if len(years) < 3:
        raise ValueError("At least three elections are needed to build two lags.")
    wide = turnout.pivot(
        index="id_brut_miom", columns="election_year", values="turnout"
    )
    frames = []
    for position in range(2, len(years)):
        year = years[position]
        previous = years[position - 1]
        before_previous = years[position - 2]
        current = turnout[turnout["election_year"] == year].set_index("id_brut_miom")
        frame = pd.DataFrame(
            {
                "election_year": year,
                "prev_turnout": wide.loc[current.index, previous],
                "prev2_turnout": wide.loc[current.index, before_previous],
                "registered": current["registered"],
                "dept_code": current["dept_code"],
                "turnout": current["turnout"],
            }
        )
        frames.append(frame.dropna(subset=["prev_turnout", "prev2_turnout"]))
    return pd.concat(frames).reset_index()


def _bucket(station_id: str) -> int:
    """Stable bucket of a polling station (same result on every machine)."""
    return zlib.crc32(station_id.encode("utf-8")) % HOLDOUT_SHARE


def split_train_eval(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one station out of five of the latest election for evaluation."""
    latest = table["election_year"].max()
    in_bucket = table["id_brut_miom"].map(_bucket) == 0
    is_eval = (table["election_year"] == latest) & in_bucket
    return table[~is_eval], table[is_eval]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the modelling table.")
    parser.add_argument("--raw", default="data/general_results.parquet")
    parser.add_argument("--max-year", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    table = build_modeling_table(load_legislative_turnout(args.raw), args.max_year)
    table.to_parquet(args.out, index=False)
    print(f"{len(table)} rows written to {args.out} (up to {args.max_year})")


if __name__ == "__main__":
    main()
