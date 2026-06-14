"""Hyperparameter tuning by walk-forward RPS.

Stage 1 (expensive): sweep DC time-decay xi x friendly-weight fw. Each rebuilds the walk-forward
feature table, trains the stacker per test year, and scores. The stacker<->DC blend is swept
cheaply on top (pure-xgb predictions are reused across blend values).

Primary metric: aggregate RPS over the test window (lower = better); ties broken by log-loss.
"""
import itertools
import time

import numpy as np
import pandas as pd

from data import load_results, played_matches
from features import build_training_table, FEATURE_COLS
from stacker import Stacker
from backtest import rps_mean, logloss, accuracy

XI_GRID = [0.0012, 0.0018, 0.0024, 0.0030]
FW_GRID = [0.5, 0.7, 1.0]
BLEND_GRID = [0.0, 0.15, 0.30, 0.45, 0.60]
TEST_YEARS = range(2021, 2026)


def eval_table(tbl):
    """Return per-test-year stacked (pure-xgb) probs and dc probs + labels, concatenated."""
    P_xgb, P_dc, Y = [], [], []
    for yr in TEST_YEARS:
        tr = tbl[tbl.date < f"{yr}-01-01"]
        te = tbl[(tbl.date >= f"{yr}-01-01") & (tbl.date < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 500:
            continue
        s = Stacker(blend_dc=0.0).fit(tr)          # pure xgb
        P_xgb.append(s._xgb_proba(te[FEATURE_COLS].values.astype(float)))
        P_dc.append(te[["dc_p_home", "dc_p_draw", "dc_p_away"]].values)
        Y.append(te["result"].values.astype(int))
    return np.vstack(P_xgb), np.vstack(P_dc), np.concatenate(Y)


def main():
    pm = played_matches(load_results())
    results = []
    t0 = time.time()
    for xi, fw in itertools.product(XI_GRID, FW_GRID):
        tb = time.time()
        tbl, _ = build_training_table(pm, dc_time_decay=xi, dc_friendly_weight=fw, verbose=False)
        p_xgb, p_dc, y = eval_table(tbl)
        for b in BLEND_GRID:
            p = (1 - b) * p_xgb + b * p_dc
            p = p / p.sum(1, keepdims=True)
            results.append(dict(xi=xi, fw=fw, blend=b, rps=rps_mean(p, y),
                                logloss=logloss(p, y), acc=accuracy(p, y)))
        # DC-only reference for this (xi,fw)
        dc_only = dict(xi=xi, fw=fw, blend="DC-only", rps=rps_mean(p_dc, y),
                       logloss=logloss(p_dc, y), acc=accuracy(p_dc, y))
        results.append(dc_only)
        best_here = min([r for r in results if r["xi"] == xi and r["fw"] == fw
                         and r["blend"] != "DC-only"], key=lambda r: r["rps"])
        print(f"xi={xi:.4f} fw={fw:.2f} | best blend={best_here['blend']:.2f} "
              f"rps={best_here['rps']:.5f} ll={best_here['logloss']:.4f} acc={best_here['acc']:.4f} "
              f"| dc-only rps={dc_only['rps']:.5f}  ({time.time()-tb:.0f}s)")

    res = pd.DataFrame([r for r in results if r["blend"] != "DC-only"])
    res = res.sort_values("rps").reset_index(drop=True)
    print(f"\n=== TOP 8 CONFIGS (by RPS) — total {time.time()-t0:.0f}s ===")
    print(res.head(8).to_string(index=False))
    best = res.iloc[0]
    print(f"\nBEST: xi={best.xi} friendly_weight={best.fw} blend_dc={best.blend} "
          f"-> RPS={best.rps:.5f} logloss={best.logloss:.4f} acc={best.acc:.4f}")
    from config import OUTPUTS
    res.to_csv(OUTPUTS / "tuning_results.csv", index=False)
    print(f"saved {OUTPUTS/'tuning_results.csv'}")


if __name__ == "__main__":
    main()
