"""Walk-forward validation of the feature-aware ScoreEngine vs the Dixon-Coles baseline, on the two
outputs it targets: exact-score hit-rate and half-time-lead accuracy (plus grid-implied 1X2 RPS)."""
import numpy as np
import pandas as pd

from config import DATA_PROC, FIRST_HALF_FRACTION
from backtest import build_ht_actuals, rps_mean
from score_engine import ScoreEngine, bivariate_grid
from first_half import ht_probs_from_lambdas


def run(test_years=range(2021, 2026)):
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl").dropna(subset=["result"]).reset_index(drop=True)
    ht = build_ht_actuals()
    ht_key = ht.set_index(["date", "home_team", "away_team"])["ht_leader"].to_dict()

    eng_exact = dc_exact = n = 0
    eng_ht = thin_ht = ht_n = 0
    eng_p, dc_p, Y = [], [], []

    for yr in test_years:
        tr = tbl[tbl.date < f"{yr}-01-01"]
        te = tbl[(tbl.date >= f"{yr}-01-01") & (tbl.date < f"{yr+1}-01-01")]
        if len(te) == 0 or len(tr) < 500:
            continue
        ht_tr = ht[ht.date < f"{yr}-01-01"]
        eng = ScoreEngine().fit(tr, ht_tr)
        for d in te.to_dict("records"):
            lh, la, lhf, laf = eng.lambdas(d)
            grid = bivariate_grid(lh, la)
            ml = np.unravel_index(np.argmax(grid), grid.shape)
            ah, aa = int(d["home_score"]), int(d["away_score"])
            eng_exact += int(ml == (ah, aa))
            dc_exact += int((d["dc_ml_home"], d["dc_ml_away"]) == (ah, aa))
            n += 1
            X, Yi = np.indices(grid.shape)
            eng_p.append([grid[X > Yi].sum(), grid[X == Yi].sum(), grid[X < Yi].sum()])
            dc_p.append([d["dc_p_home"], d["dc_p_draw"], d["dc_p_away"]])
            Y.append(int(d["result"]))
            key = (d["date"], d["home_team"], d["away_team"])
            if key in ht_key:
                # engine dedicated first-half lambdas
                g2 = bivariate_grid(lhf, laf, kmax=8)
                Xx, Yy = np.indices(g2.shape)
                eng_lead = int(np.argmax([g2[Xx > Yy].sum(), g2[Xx == Yy].sum(), g2[Xx < Yy].sum()]))
                # thinning baseline from full DC lambdas
                ph, pl, pa = ht_probs_from_lambdas(d["dc_lambda_home"], d["dc_lambda_away"],
                                                   f=FIRST_HALF_FRACTION)
                thin_lead = int(np.argmax([ph, pl, pa]))
                eng_ht += int(eng_lead == ht_key[key])
                thin_ht += int(thin_lead == ht_key[key])
                ht_n += 1
        print(f"  {yr}: done ({len(te)} matches)")

    eng_p = np.array(eng_p); dc_p = np.array(dc_p); Y = np.array(Y)
    print("\n================ ScoreEngine vs Dixon-Coles baseline ================")
    print(f"matches: {n}  | HT-covered: {ht_n}")
    print(f"{'':22s}{'ScoreEngine':>14s}{'DC baseline':>14s}")
    print(f"{'exact-score hit':22s}{eng_exact/n:>14.4f}{dc_exact/n:>14.4f}")
    print(f"{'HT-lead accuracy':22s}{eng_ht/ht_n:>14.4f}{thin_ht/ht_n:>14.4f}  (DC=Poisson thinning f=0.447)")
    print(f"{'grid 1X2 RPS':22s}{rps_mean(eng_p,Y):>14.4f}{rps_mean(dc_p,Y):>14.4f}")


if __name__ == "__main__":
    run()
