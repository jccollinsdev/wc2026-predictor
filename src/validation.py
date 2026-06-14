"""Validation suite: backtest the FINAL model on the real 2018 & 2022 World Cups (leakage-free —
stacker + score engine trained only on pre-tournament data) and benchmark vs the bookmaker market
(football-data.co.uk average closing odds, vig-removed).
"""
import unicodedata

import numpy as np
import pandas as pd

from config import DATA_PROC, DATA_RAW
from backtest import build_ht_actuals, rps_mean, logloss, accuracy
from stacker import Stacker
from score_engine import ScoreEngine, bivariate_grid

W_ENGINE = 0.5
ALIAS = {"korea republic": "south korea", "china pr": "china pr", "ir iran": "iran",
         "czechia": "czech republic", "usa": "united states", "turkiye": "turkey"}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", "and").replace(".", "").strip()
    return ALIAS.get(s, s)


def _market_probs(row):
    o = np.array([row["H-Avg"], row["D-Avg"], row["A-Avg"]], float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1):
        return None
    imp = 1.0 / o
    return imp / imp.sum()


def validate_tournament(tbl, sheet, start, end, label):
    eng_ht_actuals = build_ht_actuals()
    tr = tbl[tbl.date < start]
    stk = Stacker().fit(tr)
    eng = ScoreEngine().fit(tr, eng_ht_actuals[eng_ht_actuals.date < start])

    odds = pd.ExcelFile(DATA_RAW / "WorldCup2026.xlsx").parse(sheet)
    # index the model's feature table by normalized (home, away) within the tournament window
    win = tbl[(tbl.date >= start) & (tbl.date <= end)]
    tindex = {(_norm(r["home_team"]), _norm(r["away_team"])): r for r in win.to_dict("records")}

    PM, PK, Y = [], [], []   # model, market, actual (matched subset)
    PM_all, Y_all = [], []
    for _, o in odds.iterrows():
        key = (_norm(o["Home"]), _norm(o["Away"]))
        d = tindex.get(key)
        if d is None:
            continue
        ps = np.asarray(stk.predict_proba_row(d))   # calibrated stacker 1X2 (DC-blended internally)
        lh, la, _, _ = eng.lambdas(d)
        g = bivariate_grid(lh, la); X, Yi = np.indices(g.shape)
        gp = np.array([g[X > Yi].sum(), g[X == Yi].sum(), g[X < Yi].sum()])
        pm = W_ENGINE * gp + (1 - W_ENGINE) * ps; pm /= pm.sum()
        y = int(d["result"])
        PM_all.append(pm); Y_all.append(y)
        mp = _market_probs(o)
        if mp is not None:
            PM.append(pm); PK.append(mp); Y.append(y)
    PM, PK, Y = np.array(PM), np.array(PK), np.array(Y)
    PM_all, Y_all = np.array(PM_all), np.array(Y_all)
    print(f"\n=== {label} ===  ({len(Y_all)} matches, {len(Y)} with market odds)")
    print(f"  MODEL : acc {accuracy(PM_all,Y_all):.3f}  RPS {rps_mean(PM_all,Y_all):.4f}  logloss {logloss(PM_all,Y_all):.3f}")
    if len(Y):
        print(f"  vs MARKET (matched subset):")
        print(f"     model  : acc {accuracy(PM,Y):.3f}  RPS {rps_mean(PM,Y):.4f}")
        print(f"     market : acc {accuracy(PK,Y):.3f}  RPS {rps_mean(PK,Y):.4f}   <- bookmaker closing odds")
    return PM_all, Y_all


if __name__ == "__main__":
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl").dropna(subset=["result"]).reset_index(drop=True)
    a = validate_tournament(tbl, "WorldCup2018", pd.Timestamp("2018-06-01"), pd.Timestamp("2018-07-31"),
                            "WORLD CUP 2018 (leakage-free)")
    b = validate_tournament(tbl, "WorldCup2022", pd.Timestamp("2022-11-01"), pd.Timestamp("2022-12-31"),
                            "WORLD CUP 2022 (leakage-free)")
    print("\n(Beating the market is not expected — closing odds are the gold standard. Being CLOSE validates the model.)")
