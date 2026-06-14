"""Leakage-free retrodiction scorecard: what the model WOULD have predicted for already-played
WC2026 games (using only data before each kickoff, stacker retrained on prior matches only) vs what
actually happened.
"""
import numpy as np
import pandas as pd

from config import DATA_PROC
from data import load_results
from predict import Predictor
from stacker import Stacker
from score_engine import ScoreEngine
from backtest import rps_mean, build_ht_actuals

OUT = ["home", "draw", "away"]


def run():
    df = load_results()
    wc_played = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-01-01") & df.played] \
        .sort_values("date")
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl")

    pred_cache = {}
    rows = []
    probs_all, y_all = [], []
    for _, m in wc_played.iterrows():
        asof = m["date"]
        key = asof.isoformat()
        if key not in pred_cache:
            p = Predictor(asof=asof, verbose=False)
            pre = tbl[tbl.date < asof]
            p.stacker = Stacker().fit(pre)                                   # leakage-free stacker
            _ht = build_ht_actuals()
            p.engine = ScoreEngine().fit(pre, _ht[_ht.date < asof])          # leakage-free engine
            pred_cache[key] = p
        P = pred_cache[key]
        r = P.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]), date=asof)
        actual = (0 if m["home_score"] > m["away_score"]
                  else 1 if m["home_score"] == m["away_score"] else 2)
        probs = np.array([r["p_home"], r["p_draw"], r["p_away"]])
        probs_all.append(probs); y_all.append(actual)
        rows.append(dict(
            date=m["date"].date(), match=f'{m["home_team"]} v {m["away_team"]}',
            pred=f"{r['headline_score'][0]}-{r['headline_score'][1]} ({r['outcome']})",
            actual=f'{int(m["home_score"])}-{int(m["away_score"])} ({OUT[actual]})',
            P_home=round(r["p_home"], 2), P_draw=round(r["p_draw"], 2), P_away=round(r["p_away"], 2),
            outcome_hit="Y" if OUT.index(r["outcome"]) == actual else "n",
            HT_pred=r["ht"]["ht_leader"],
        ))
    res = pd.DataFrame(rows)
    P = np.vstack(probs_all); Y = np.array(y_all)
    print(res.to_string(index=False))
    print(f"\noutcome accuracy: {(res.outcome_hit=='Y').mean():.0%} ({(res.outcome_hit=='Y').sum()}/{len(res)})"
          f"   |   mean RPS: {rps_mean(P,Y):.4f}   (lower=better; ~0.20 random)")
    print("note: 4 games is a tiny sample — single-match calls are inherently noisy.")
    return res


if __name__ == "__main__":
    run()
