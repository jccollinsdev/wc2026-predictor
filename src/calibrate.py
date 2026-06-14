"""RPS-targeted, regime-aware probability calibration.

The stacker trains on log-loss but we're judged on RPS (ordinal). On real World Cups the market beats
us only on RPS, suggesting tail miscalibration in the high-stakes regime. We fit a temperature per
regime (major tournament: tier>=4, vs other) that MINIMIZES RPS on a time-split, then save it.
Applied as p' ∝ p**(1/T) to the final blended 1X2.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from config import DATA_PROC, MODELS
from backtest import build_ht_actuals, rps_mean, accuracy
from stacker import Stacker
from score_engine import ScoreEngine, bivariate_grid

W_ENGINE = 0.5


def _temp(p, T):
    q = np.clip(p, 1e-9, 1) ** (1.0 / T)
    return q / q.sum(1, keepdims=True)


def _best_T(P, Y):
    r = minimize_scalar(lambda T: rps_mean(_temp(P, T), Y), bounds=(0.5, 3.0), method="bounded")
    return float(r.x)


def collect(test_years):
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl").dropna(subset=["result"]).reset_index(drop=True)
    ht = build_ht_actuals()
    P, Y, TIER, YR = [], [], [], []
    for yr in test_years:
        tr = tbl[tbl.date < f"{yr}-01-01"]
        te = tbl[(tbl.date >= f"{yr}-01-01") & (tbl.date < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 500:
            continue
        stk = Stacker().fit(tr)
        eng = ScoreEngine().fit(tr, ht[ht.date < f"{yr}-01-01"])
        ps = stk.predict_proba(te)
        for i, d in enumerate(te.to_dict("records")):
            lh, la, _, _ = eng.lambdas(d)
            g = bivariate_grid(lh, la); X, Yi = np.indices(g.shape)
            gp = np.array([g[X > Yi].sum(), g[X == Yi].sum(), g[X < Yi].sum()])
            p = W_ENGINE * gp + (1 - W_ENGINE) * ps[i]; p /= p.sum()
            P.append(p); Y.append(int(d["result"])); TIER.append(int(d["tier"])); YR.append(yr)
        print(f"  collected {yr}")
    return np.array(P), np.array(Y), np.array(TIER), np.array(YR)


def main():
    P, Y, TIER, YR = collect(range(2019, 2026))
    fit = YR <= 2023
    tst = YR >= 2024
    major = TIER >= 4
    # global RPS-optimal temperature (fit on <=2023)
    Tg = _best_T(P[fit], Y[fit])
    # regime temps
    Tm = _best_T(P[fit & major], Y[fit & major]) if (fit & major).sum() > 50 else Tg
    To = _best_T(P[fit & ~major], Y[fit & ~major]) if (fit & ~major).sum() > 50 else Tg

    def apply_regime(P, TIER):
        out = P.copy()
        m = TIER >= 4
        out[m] = _temp(P[m], Tm); out[~m] = _temp(P[~m], To)
        return out

    print(f"\nfit temperatures -> global {Tg:.3f} | major(tier>=4) {Tm:.3f} | other {To:.3f}")
    print(f"\n{'(test 2024-25)':22s}{'RPS':>9s}{'acc':>8s}")
    print(f"{'raw blend':22s}{rps_mean(P[tst],Y[tst]):>9.4f}{accuracy(P[tst],Y[tst]):>8.3f}")
    print(f"{'global temp':22s}{rps_mean(_temp(P[tst],Tg),Y[tst]):>9.4f}{accuracy(_temp(P[tst],Tg),Y[tst]):>8.3f}")
    pr = apply_regime(P, TIER)
    print(f"{'regime temp':22s}{rps_mean(pr[tst],Y[tst]):>9.4f}{accuracy(pr[tst],Y[tst]):>8.3f}")
    # also report on the major-tournament subset of test (closest analogue to WC)
    tm = tst & major
    if tm.sum() > 20:
        print(f"\n  on MAJOR-tournament test matches (n={tm.sum()}):")
        print(f"     raw   RPS {rps_mean(P[tm],Y[tm]):.4f}")
        print(f"     regime RPS {rps_mean(_temp(P[tm],Tm),Y[tm]):.4f}")

    # refit on ALL data for production and save
    Tg_all = _best_T(P, Y)
    Tm_all = _best_T(P[major], Y[major]) if major.sum() > 50 else Tg_all
    To_all = _best_T(P[~major], Y[~major]) if (~major).sum() > 50 else Tg_all
    cal = dict(T_major=Tm_all, T_other=To_all, T_global=Tg_all)
    (MODELS / "calibrator.json").write_text(json.dumps(cal, indent=2))
    print(f"\nsaved calibrator -> {MODELS/'calibrator.json'}: {cal}")


if __name__ == "__main__":
    main()
