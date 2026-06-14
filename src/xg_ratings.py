"""xG-based team form/strength from the eatpizzanot international xG dataset (2017-2026, ~4,973
senior international matches with home/away xG). Expected goals are less noisy than goals, so rolling
xG-for / xG-against are a cleaner strength signal than goal-based form.

Leakage-free as-of-date rolling features, keyed to martj42 team names.
"""
import bisect
import functools
import unicodedata

import numpy as np
import pandas as pd

from config import DATA_RAW

# xG-dataset name -> martj42 name (only mismatches; rest matched by normalization)
ALIAS = {"usa": "united states", "korea republic": "south korea", "czechia": "czech republic",
         "turkiye": "turkey", "china": "china pr", "ir iran": "iran", "cape verde islands": "cape verde",
         "congo dr": "dr congo"}
_YOUTH = ("u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23", " b ", " olympic")
XG_COLS = ["xg_for_roll", "xg_against_roll"]
ROLL_N = 10


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and").replace(".", "").strip()
    s = " ".join(s.split())
    return ALIAS.get(s, s)


def _is_senior(name):
    n = " " + str(name).lower() + " "
    return not any(y in n for y in _YOUTH)


@functools.lru_cache(maxsize=1)
def _history():
    """{normalized team -> (sorted date array, xg_for array, xg_against array)}."""
    fx = pd.read_csv(DATA_RAW / "fixtures.csv", low_memory=False)
    ms = pd.read_csv(DATA_RAW / "match_stats.csv", low_memory=False)[["fixture_id", "home_xg", "away_xg"]]
    tm = pd.read_csv(DATA_RAW / "teams_xg.csv", low_memory=False)
    id2name = dict(zip(tm.id, tm.name))
    d = fx[fx.league_id.between(78, 88)].merge(ms, left_on="id", right_on="fixture_id", how="inner")
    d = d.dropna(subset=["home_xg", "away_xg"]).copy()
    d["home"] = d.home_team_id.map(id2name)
    d["away"] = d.away_team_id.map(id2name)
    d = d.dropna(subset=["home", "away"])
    d = d[d.home.map(_is_senior) & d.away.map(_is_senior)]
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    d = d.dropna(subset=["date"]).sort_values("date")

    tmp = {}
    for r in d.itertuples(index=False):
        h, a = _norm(r.home), _norm(r.away)
        tmp.setdefault(h, []).append((r.date, float(r.home_xg), float(r.away_xg)))
        tmp.setdefault(a, []).append((r.date, float(r.away_xg), float(r.home_xg)))
    out = {}
    for t, lst in tmp.items():
        lst.sort()
        out[t] = (np.array([x[0] for x in lst], dtype="datetime64[ns]"),
                  np.array([x[1] for x in lst]), np.array([x[2] for x in lst]))
    return out


def xg_features(team, as_of, n=ROLL_N):
    hist = _history()
    key = _norm(team)
    if key not in hist:
        return {"xg_for_roll": np.nan, "xg_against_roll": np.nan}
    dates, xf, xa = hist[key]
    i = bisect.bisect_left(dates, np.datetime64(pd.Timestamp(as_of)))
    if i < 3:
        return {"xg_for_roll": np.nan, "xg_against_roll": np.nan}
    lo = max(0, i - n)
    return {"xg_for_roll": float(xf[lo:i].mean()), "xg_against_roll": float(xa[lo:i].mean())}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from tournament import GROUPS
    wc = sorted({t for ts in GROUPS.values() for t in ts})
    have = [t for t in wc if _norm(t) in _history()]
    print(f"xG coverage of WC2026 teams: {len(have)}/48  (missing: {[t for t in wc if t not in have]})")
    for t in ["United States", "Paraguay", "Brazil", "Spain"]:
        f = xg_features(t, "2026-06-13")
        print(f"  {t:16s} rolling xG for {f['xg_for_roll']:.2f}  against {f['xg_against_roll']:.2f}")
