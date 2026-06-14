"""Load and clean the martj42 international football datasets.

Single source of truth -> team names are internally consistent (no cross-source mapping needed
for the core model). Handles the data quirks observed in the raw files:
  - away/home score '00' (e.g. Mexico 2-00 South Africa) -> coerced to int via to_numeric
  - 'NA' scores -> unplayed fixtures (kept separately for prediction targets)
  - goal minutes like '45+2' -> parsed to first/second half for the first-half fraction
"""
import numpy as np
import pandas as pd

from config import RESULTS_CSV, GOALSCORERS_CSV, SHOOTOUTS_CSV


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    # coerce scores; '00' -> 0, 'NA'/'' -> NaN (unplayed)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # outcome label for played matches: 0 home win, 1 draw, 2 away win
    res = np.full(len(df), -1, dtype=int)
    pm = df["played"].values
    hs = df["home_score"].values
    as_ = df["away_score"].values
    res[pm & (hs > as_)] = 0
    res[pm & (hs == as_)] = 1
    res[pm & (hs < as_)] = 2
    df["result"] = res
    return df


def played_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["played"]].copy().reset_index(drop=True)


def upcoming_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["played"]].copy().reset_index(drop=True)


def _parse_minute(m) -> float:
    """'45+2' -> 47-ish but for half assignment we only need the base minute; '45+x' is 1st half."""
    if pd.isna(m):
        return np.nan
    s = str(m).strip()
    if "+" in s:
        base = s.split("+")[0]
        try:
            return float(base)  # 45+ -> first-half stoppage counts as first half
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def first_half_fraction() -> float:
    """Empirical fraction of goals scored in the first half (minute<=45, incl. 45+ stoppage)."""
    g = pd.read_csv(GOALSCORERS_CSV)
    mins = g["minute"].map(_parse_minute).dropna()
    fh = (mins <= 45).sum()
    return float(fh / len(mins))


def load_shootouts() -> pd.DataFrame:
    s = pd.read_csv(SHOOTOUTS_CSV)
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    return s


if __name__ == "__main__":
    df = load_results()
    pm = played_matches(df)
    up = upcoming_matches(df)
    print(f"total rows           : {len(df):,}")
    print(f"played matches       : {len(pm):,}")
    print(f"upcoming fixtures     : {len(up):,}")
    print(f"date range           : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"distinct teams        : {pd.concat([df.home_team, df.away_team]).nunique():,}")
    print(f"first-half fraction   : {first_half_fraction():.4f}")
    print("\nUSA vs Paraguay upcoming row:")
    mask = (up.home_team.isin(["United States", "Paraguay"]) &
            up.away_team.isin(["United States", "Paraguay"]))
    print(up[mask][["date", "home_team", "away_team", "tournament", "city", "country", "neutral"]].to_string(index=False))
    print("\nWC2026 fixtures total:", (up.tournament == "FIFA World Cup").sum())
