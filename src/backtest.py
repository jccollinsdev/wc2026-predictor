"""Walk-forward, time-based backtest. No leakage: for each test year, train the stacker only on
prior matches; DC features are already walk-forward; Elo is pre-match by construction.

Metrics: RPS (primary, ordinal 1X2), log-loss, 1X2 accuracy, exact-score hit-rate, and half-time
lead accuracy (vs actual minute<=45 goals reconstructed from goalscorers.csv).
Baseline: Dixon-Coles alone (the dc_p_* columns).
"""
import numpy as np
import pandas as pd

from config import DATA_PROC, GOALSCORERS_CSV, FIRST_HALF_FRACTION
from features import FEATURE_COLS
from stacker import Stacker
from first_half import ht_probs_from_lambdas
from data import _parse_minute


# ---------- metrics ----------
def rps_1x2(p, y):
    """Ranked Probability Score for ordered outcomes home(0)<draw(1)<away(2). Lower better."""
    p = np.asarray(p, float)
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1.0
    cp = np.cumsum(p, axis=1)[:, :2]
    co = np.cumsum(o, axis=1)[:, :2]
    return np.mean(np.sum((cp - co) ** 2, axis=1))  # /(r-1)=/2 ... use standard 0.5 factor below


def rps_mean(p, y):
    p = np.asarray(p, float)
    o = np.zeros((len(y), 3))
    o[np.arange(len(y)), y] = 1.0
    cp = np.cumsum(p, axis=1)
    co = np.cumsum(o, axis=1)
    return np.mean(0.5 * np.sum((cp - co) ** 2, axis=1))


def logloss(p, y):
    p = np.asarray(p, float)
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))


def accuracy(p, y):
    return float((np.asarray(p).argmax(1) == y).mean())


# ---------- actual half-time scores from goalscorers ----------
def build_ht_actuals():
    """Reconstruct HT scores for matches with FULL goal-minute coverage.

    Critically includes matches that were 0-0 at half-time (which have no first-half goal rows):
    a played match is 'covered' iff the number of goals with a valid minute equals its full-time
    total (so 0-0 FT and 'all goals 2nd half' both resolve correctly to level-at-HT).
    """
    from data import load_results, played_matches
    pm = played_matches(load_results())
    g = pd.read_csv(GOALSCORERS_CSV)
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g["m"] = g["minute"].map(_parse_minute)
    gv = g.dropna(subset=["m"]).copy()
    gv["is_home_goal"] = gv["team"] == gv["home_team"]
    gv["fh"] = gv["m"] <= 45
    gv["fh_home"] = gv["fh"] & gv["is_home_goal"]
    gv["fh_away"] = gv["fh"] & (~gv["is_home_goal"])
    gg = gv.groupby(["date", "home_team", "away_team"]).agg(
        n_valid=("m", "size"), hth=("fh_home", "sum"), hta=("fh_away", "sum")).reset_index()
    m = pm.merge(gg, on=["date", "home_team", "away_team"], how="left")
    m["n_valid"] = m["n_valid"].fillna(0)
    m["hth"] = m["hth"].fillna(0).astype(int)
    m["hta"] = m["hta"].fillna(0).astype(int)
    m["ft_total"] = m["home_score"] + m["away_score"]
    cov = m[m["n_valid"] == m["ft_total"]].copy()   # full minute coverage (incl. 0-0 FT)
    cov["ht_leader"] = np.where(cov.hth > cov.hta, 0, np.where(cov.hth == cov.hta, 1, 2))
    return cov[["date", "home_team", "away_team", "hth", "hta", "ht_leader"]]


def backtest_full(test_years=range(2019, 2026), verbose=True):
    """Final model: stacker (calibrated 1X2) blended with the feature-aware ScoreEngine grid-1X2,
    scorelines + exact-score from the engine grid, half-time from the engine's first-half model.
    Reports each component and sweeps the blend weight."""
    from score_engine import ScoreEngine, bivariate_grid
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl").dropna(subset=["result"]).reset_index(drop=True)
    ht = build_ht_actuals()
    ht_key = ht.set_index(["date", "home_team", "away_team"])["ht_leader"].to_dict()

    PS, PE, PDC, Y = [], [], [], []
    eng_exact = dc_exact = n = 0
    eng_ht = thin_ht = ht_n = 0
    for yr in test_years:
        tr = tbl[tbl.date < f"{yr}-01-01"]
        te = tbl[(tbl.date >= f"{yr}-01-01") & (tbl.date < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 500:
            continue
        stk = Stacker().fit(tr)
        eng = ScoreEngine().fit(tr, ht[ht.date < f"{yr}-01-01"])
        ps = stk.predict_proba(te)
        for i, d in enumerate(te.to_dict("records")):
            lh, la, lhf, laf = eng.lambdas(d)
            grid = bivariate_grid(lh, la)
            X, Yi = np.indices(grid.shape)
            PE.append([grid[X > Yi].sum(), grid[X == Yi].sum(), grid[X < Yi].sum()])
            PS.append(ps[i]); PDC.append([d["dc_p_home"], d["dc_p_draw"], d["dc_p_away"]])
            Y.append(int(d["result"]))
            ml = np.unravel_index(np.argmax(grid), grid.shape)
            ah, aa = int(d["home_score"]), int(d["away_score"])
            eng_exact += int(ml == (ah, aa)); dc_exact += int((d["dc_ml_home"], d["dc_ml_away"]) == (ah, aa)); n += 1
            key = (d["date"], d["home_team"], d["away_team"])
            if key in ht_key:
                g2 = bivariate_grid(lhf, laf, kmax=8); Xx, Yy = np.indices(g2.shape)
                el = int(np.argmax([g2[Xx > Yy].sum(), g2[Xx == Yy].sum(), g2[Xx < Yy].sum()]))
                from first_half import ht_probs_from_lambdas
                ph, pl, pa = ht_probs_from_lambdas(d["dc_lambda_home"], d["dc_lambda_away"])
                tl = int(np.argmax([ph, pl, pa]))
                eng_ht += int(el == ht_key[key]); thin_ht += int(tl == ht_key[key]); ht_n += 1
        if verbose:
            print(f"  {yr}: {len(te)} matches")
    PS = np.array(PS); PE = np.array(PE); PDC = np.array(PDC); Y = np.array(Y)
    # sweep blend weight (engine grid weight)
    best_w, best_rps = 0.0, 9
    for w in np.linspace(0, 1, 11):
        p = (1 - w) * PS + w * PE
        r = rps_mean(p / p.sum(1, keepdims=True), Y)
        if r < best_rps:
            best_rps, best_w = r, w
    pf = (1 - best_w) * PS + best_w * PE; pf /= pf.sum(1, keepdims=True)
    print("\n============ FINAL MODEL (stacker + ScoreEngine blend) ============")
    print(f"matches: {len(Y)}  | HT-covered: {ht_n}")
    print(f"{'component':24s}{'acc':>8s}{'RPS':>9s}{'logloss':>9s}")
    for name, P in [("stacker only", PS), ("engine-grid only", PE), ("DC only", PDC),
                    (f"BLEND (w_engine={best_w:.1f})", pf)]:
        print(f"{name:24s}{accuracy(P,Y):>8.3f}{rps_mean(P,Y):>9.4f}{logloss(P,Y):>9.3f}")
    print(f"\n{'exact-score':24s}engine {eng_exact/n:.4f}  vs DC {dc_exact/n:.4f}")
    print(f"{'HT-lead acc':24s}engine {eng_ht/ht_n:.4f}  vs thinning {thin_ht/ht_n:.4f}")
    return dict(best_w=best_w, P=pf, Y=Y)


def run_backtest(test_years=range(2019, 2026), verbose=True):
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl")
    tbl = tbl.dropna(subset=["result"]).reset_index(drop=True)
    ht = build_ht_actuals()
    ht_key = ht.set_index(["date", "home_team", "away_team"])["ht_leader"].to_dict()

    all_p, all_pdc, all_y = [], [], []
    exact_hit, exact_n = 0, 0
    ht_correct, ht_n = 0, 0
    ht_actuals_list = []

    for yr in test_years:
        tr = tbl[tbl.date < f"{yr}-01-01"]
        te = tbl[(tbl.date >= f"{yr}-01-01") & (tbl.date < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 500:
            continue
        s = Stacker().fit(tr)
        p = s.predict_proba(te)
        pdc = te[["dc_p_home", "dc_p_draw", "dc_p_away"]].values
        y = te["result"].values.astype(int)
        all_p.append(p); all_pdc.append(pdc); all_y.append(y)

        # exact score (DC modal scoreline vs actual)
        ml_h = te["dc_ml_home"].values; ml_a = te["dc_ml_away"].values
        ah = te["home_score"].values; aa = te["away_score"].values
        exact_hit += int(np.sum((ml_h == ah) & (ml_a == aa)))
        exact_n += len(te)

        # HT lead accuracy
        for _, r in te.iterrows():
            key = (r["date"], r["home_team"], r["away_team"])
            if key in ht_key:
                ph, pl, pa = ht_probs_from_lambdas(r["dc_lambda_home"], r["dc_lambda_away"],
                                                   f=FIRST_HALF_FRACTION)
                pred_leader = int(np.argmax([ph, pl, pa]))
                ht_correct += int(pred_leader == ht_key[key])
                ht_n += 1
                ht_actuals_list.append(ht_key[key])
        if verbose:
            print(f"  {yr}: n={len(te):4d}  acc={accuracy(p,y):.3f}  rps={rps_mean(p,y):.4f}  "
                  f"logloss={logloss(p,y):.3f}")

    P = np.vstack(all_p); PDC = np.vstack(all_pdc); Y = np.concatenate(all_y)
    print("\n================ AGGREGATE (walk-forward) ================")
    print(f"matches evaluated : {len(Y):,}")
    print(f"{'':16s}{'STACKER':>12s}{'DC-only':>12s}")
    print(f"{'accuracy':16s}{accuracy(P,Y):>12.3f}{accuracy(PDC,Y):>12.3f}")
    print(f"{'RPS':16s}{rps_mean(P,Y):>12.4f}{rps_mean(PDC,Y):>12.4f}")
    print(f"{'log-loss':16s}{logloss(P,Y):>12.3f}{logloss(PDC,Y):>12.3f}")
    print(f"{'exact-score':16s}{exact_hit/exact_n:>12.3f}   (DC modal scoreline, n={exact_n})")
    hta = np.array(ht_actuals_list)
    ht_base = np.bincount(hta, minlength=3).max() / len(hta)
    print(f"{'HT-lead acc':16s}{ht_correct/ht_n:>12.3f}   (n={ht_n}; majority-class baseline {ht_base:.3f})")
    print(f"   actual HT split among evaluated: home-lead {(hta==0).mean():.3f} / "
          f"level {(hta==1).mean():.3f} / away-lead {(hta==2).mean():.3f}")
    # naive baselines
    base = np.tile(np.bincount(Y, minlength=3) / len(Y), (len(Y), 1))
    print(f"\nFT baseline (always predict class priors): acc={accuracy(base,Y):.3f} "
          f"rps={rps_mean(base,Y):.4f} logloss={logloss(base,Y):.3f}")
    return dict(P=P, Y=Y)


if __name__ == "__main__":
    run_backtest()
