"""XGBoost 1X2 stacker: meta-learner over DC probs + Elo + form/rest/venue features.

Because the Dixon-Coles implied probabilities are *inputs* to the model, the XGBoost output is the
stacked prediction. We additionally blend with raw DC probs for stability/calibration (weight tuned
on the backtest; default 0.7 xgb / 0.3 dc).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURE_COLS

CLASSES = [0, 1, 2]  # home, draw, away


class Stacker:
    def __init__(self, blend_dc=0.45, params=None, num_round=350):
        self.blend_dc = blend_dc
        self.num_round = num_round
        self.params = params or dict(
            objective="multi:softprob", num_class=3, max_depth=4, eta=0.04,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, reg_alpha=0.0, eval_metric="mlogloss", tree_method="hist",
            nthread=0, seed=42,
        )
        self.booster = None

    def fit(self, tbl: pd.DataFrame):
        X = tbl[FEATURE_COLS].values.astype(float)
        y = tbl["result"].values.astype(int)
        dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.num_round)
        return self

    def _xgb_proba(self, X):
        d = xgb.DMatrix(X, feature_names=FEATURE_COLS)
        return self.booster.predict(d)

    def predict_proba(self, tbl: pd.DataFrame):
        X = tbl[FEATURE_COLS].values.astype(float)
        p_xgb = self._xgb_proba(X)
        p_dc = tbl[["dc_p_home", "dc_p_draw", "dc_p_away"]].values
        p = (1 - self.blend_dc) * p_xgb + self.blend_dc * p_dc
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def predict_proba_row(self, feat: dict):
        X = np.array([[feat[c] for c in FEATURE_COLS]], dtype=float)
        p_xgb = self._xgb_proba(X)[0]
        p_dc = np.array([feat["dc_p_home"], feat["dc_p_draw"], feat["dc_p_away"]])
        p = (1 - self.blend_dc) * p_xgb + self.blend_dc * p_dc
        return p / p.sum()

    def feature_importance(self):
        if self.booster is None:
            return {}
        return dict(sorted(self.booster.get_score(importance_type="gain").items(),
                           key=lambda kv: -kv[1]))


if __name__ == "__main__":
    from config import DATA_PROC
    tbl = pd.read_pickle(DATA_PROC / "train_table.pkl")
    # quick time-split sanity: train < 2024, test 2024+
    tr = tbl[tbl.date < "2024-01-01"]
    te = tbl[tbl.date >= "2024-01-01"]
    s = Stacker().fit(tr)
    p = s.predict_proba(te)
    pred = p.argmax(1)
    acc = (pred == te["result"].values).mean()
    # log loss
    yt = te["result"].values
    ll = -np.mean(np.log(np.clip(p[np.arange(len(yt)), yt], 1e-12, 1)))
    print(f"holdout 2024+: n={len(te)} acc={acc:.3f} logloss={ll:.3f}")
    print("\ntop feature importances (gain):")
    for k, v in list(s.feature_importance().items())[:12]:
        print(f"  {k:18s} {v:8.1f}")
