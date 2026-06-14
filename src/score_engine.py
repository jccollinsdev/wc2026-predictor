"""Feature-aware score engine: gradient-boosted Poisson goal models.

Dixon-Coles builds its scoreline grid from goals alone (attack/defense). This engine predicts each
team's expected goals (lambda) from the FULL feature set — squad strength, form, rest, choke index,
confederation, plus DC's own lambda as a base — so the entire score distribution (scorelines,
exact-score, half-time) becomes feature-aware, not just the 1X2 stacker.

Two XGBoost Poisson regressors on TEAM-PERSPECTIVE rows (each match -> 2 rows, symmetric):
  - full-match goals  -> lambda for the full game
  - first-half goals  -> lambda for minute<=45 (the dedicated, learned first-half model;
                          fast/slow starters, not a flat 0.45 thinning)
Bivariate scoreline grid is built from these lambdas with the Dixon-Coles tau low-score correction.
"""
import math

import numpy as np
import pandas as pd
import xgboost as xgb

_LOG_FACT = np.array([math.lgamma(k + 1) for k in range(31)])

PERSP_COLS = [
    "is_home", "is_neutral", "elo_self", "elo_opp", "elo_diff", "dc_lambda_self", "dc_lambda_opp",
    "squad_ovr_self", "squad_ovr_opp", "squad_top11_self", "squad_top11_opp",
    "squad_att_self", "squad_def_opp", "squad_def_self", "squad_att_opp", "squad_gk_self", "squad_gk_opp",
    "form_gf_self", "form_ga_self", "form_gf_opp", "form_ga_opp", "form_pts_self", "form_pts_opp",
    "rest_self", "rest_opp", "choke_self", "upset_self", "bigmatch_self", "bigmatch_opp",
    "late_goal_self", "late_concede_opp", "momentum_self", "clean_sheet_opp",
    "conf_self", "conf_opp", "cross_conf", "tier",
]   # NOTE: xG features were tested here (and in the stacker) and did NOT improve validated
    # performance — the model is at the signal ceiling — so they are intentionally excluded.


def persp_features(d, side):
    """Build a team-perspective feature dict from a match-level row dict d. side in {home, away}."""
    neu = int(d["is_neutral"])
    if side == "home":
        is_home = 0 if neu else 1
        s, o = "home", "away"
        dc_self, dc_opp = d["dc_lambda_home"], d["dc_lambda_away"]
        elo_self, elo_opp = d["elo_home_pre"], d["elo_away_pre"]
    else:
        is_home = 0
        s, o = "away", "home"
        dc_self, dc_opp = d["dc_lambda_away"], d["dc_lambda_home"]
        elo_self, elo_opp = d["elo_away_pre"], d["elo_home_pre"]
    g = lambda key: d[key]
    return {
        "is_home": is_home, "is_neutral": neu,
        "elo_self": elo_self, "elo_opp": elo_opp, "elo_diff": elo_self - elo_opp,
        "dc_lambda_self": dc_self, "dc_lambda_opp": dc_opp,
        "squad_ovr_self": g(f"{s}_squad_ovr"), "squad_ovr_opp": g(f"{o}_squad_ovr"),
        "squad_top11_self": g(f"{s}_squad_top11"), "squad_top11_opp": g(f"{o}_squad_top11"),
        "squad_att_self": g(f"{s}_squad_att"), "squad_def_opp": g(f"{o}_squad_def"),
        "squad_def_self": g(f"{s}_squad_def"), "squad_att_opp": g(f"{o}_squad_att"),
        "squad_gk_self": g(f"{s}_squad_gk"), "squad_gk_opp": g(f"{o}_squad_gk"),
        "form_gf_self": g(f"form_gf_{s}"), "form_ga_self": g(f"form_ga_{s}"),
        "form_gf_opp": g(f"form_gf_{o}"), "form_ga_opp": g(f"form_ga_{o}"),
        "form_pts_self": g(f"form_pts_{s}"), "form_pts_opp": g(f"form_pts_{o}"),
        "rest_self": g(f"rest_{s}"), "rest_opp": g(f"rest_{o}"),
        "choke_self": g(f"{s}_choke_index"), "upset_self": g(f"{s}_upset_index"),
        "bigmatch_self": g(f"{s}_bigmatch_ppg"), "bigmatch_opp": g(f"{o}_bigmatch_ppg"),
        "late_goal_self": g(f"{s}_late_goal_share"), "late_concede_opp": g(f"{o}_late_concede_share"),
        "momentum_self": g(f"{s}_elo_momentum"), "clean_sheet_opp": g(f"{o}_clean_sheet_rate"),
        "conf_self": d[f"{s}_conf"], "conf_opp": d[f"{o}_conf"], "cross_conf": d["cross_conf"],
        "tier": d["tier"],
    }


def _poisson_col(lmbda, kmax):
    k = np.arange(kmax + 1)
    return np.exp(-lmbda + k * np.log(max(lmbda, 1e-6)) - _LOG_FACT[: kmax + 1])


def bivariate_grid(lh, la, rho=-0.07, kmax=12):
    grid = np.outer(_poisson_col(lh, kmax), _poisson_col(la, kmax))
    grid[0, 0] *= (1 - lh * la * rho); grid[0, 1] *= (1 + lh * rho)
    grid[1, 0] *= (1 + la * rho); grid[1, 1] *= (1 - rho)
    grid = np.clip(grid, 0, None)
    return grid / grid.sum()


class ScoreEngine:
    def __init__(self, num_round=320, blend_dc=0.25):
        self.blend_dc = blend_dc
        self.params = dict(objective="count:poisson", max_depth=4, eta=0.05, subsample=0.85,
                           colsample_bytree=0.8, min_child_weight=6, reg_lambda=2.0,
                           tree_method="hist", max_bin=256, nthread=0, seed=7)
        self.num_round = num_round
        self.full = None
        self.fh = None

    def _persp_matrix(self, table):
        recs = table.to_dict("records")
        rows, yf, yh, yh_fh, ya_fh, has_fh = [], [], [], [], [], []
        Xh, Xa = [], []
        for d in recs:
            Xh.append(persp_features(d, "home")); Xa.append(persp_features(d, "away"))
        return recs, Xh, Xa

    def fit(self, table, ht_actuals):
        # full-match model
        recs = table.to_dict("records")
        Xrows, y = [], []
        for d in recs:
            Xrows.append(persp_features(d, "home")); y.append(d["home_score"])
            Xrows.append(persp_features(d, "away")); y.append(d["away_score"])
        X = pd.DataFrame(Xrows)[PERSP_COLS].values.astype(float)
        dtr = xgb.DMatrix(X, label=np.array(y, float), feature_names=PERSP_COLS)
        self.full = xgb.train(self.params, dtr, num_boost_round=self.num_round)

        # first-half model (covered matches only)
        ht = ht_actuals.set_index(["date", "home_team", "away_team"])[["hth", "hta"]].to_dict("index")
        Xrows, y = [], []
        for d in recs:
            key = (d["date"], d["home_team"], d["away_team"])
            if key in ht:
                Xrows.append(persp_features(d, "home")); y.append(ht[key]["hth"])
                Xrows.append(persp_features(d, "away")); y.append(ht[key]["hta"])
        X = pd.DataFrame(Xrows)[PERSP_COLS].values.astype(float)
        dtr = xgb.DMatrix(X, label=np.array(y, float), feature_names=PERSP_COLS)
        self.fh = xgb.train(self.params, dtr, num_boost_round=self.num_round)
        return self

    def _lam(self, booster, feat_dict, side):
        x = pd.DataFrame([persp_features(feat_dict, side)])[PERSP_COLS].values.astype(float)
        return float(booster.predict(xgb.DMatrix(x, feature_names=PERSP_COLS))[0])

    def lambdas(self, feat_dict):
        lh = self._lam(self.full, feat_dict, "home")
        la = self._lam(self.full, feat_dict, "away")
        # blend with DC's own lambdas for robustness
        lh = (1 - self.blend_dc) * lh + self.blend_dc * feat_dict["dc_lambda_home"]
        la = (1 - self.blend_dc) * la + self.blend_dc * feat_dict["dc_lambda_away"]
        lh_fh = self._lam(self.fh, feat_dict, "home")
        la_fh = self._lam(self.fh, feat_dict, "away")
        return lh, la, lh_fh, la_fh

    def feature_importance(self, which="full"):
        b = self.full if which == "full" else self.fh
        return dict(sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from config import DATA_PROC
    from backtest import build_ht_actuals
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl")
    eng = ScoreEngine().fit(tbl, build_ht_actuals())
    # predict the (known) USA-Paraguay row from the table
    row = tbl[(tbl.home_team == "United States") & (tbl.away_team == "Paraguay") &
              (tbl.date >= "2026-06-01")].iloc[0].to_dict()
    lh, la, lhf, laf = eng.lambdas(row)
    print(f"USA vs Paraguay feature-aware lambdas:")
    print(f"  full:  USA {lh:.2f} - {la:.2f} PAR   (DC said {row['dc_lambda_home']:.2f} - {row['dc_lambda_away']:.2f})")
    print(f"  1H:    USA {lhf:.2f} - {laf:.2f} PAR")
    print(f"  actual full-time: 4-1")
    g = bivariate_grid(lh, la)
    X, Y = np.indices(g.shape)
    print(f"  grid 1X2: home {g[X>Y].sum():.3f} draw {g[X==Y].sum():.3f} away {g[X<Y].sum():.3f}")
    print("\ntop full-model features (gain):")
    for k, v in list(eng.feature_importance().items())[:12]:
        print(f"  {k:18s} {v:8.1f}")
