"""Unified predictor: full-time score distribution + stacked 1X2 + first-half lead, for any fixture.

Usage:
  python src/predict.py                      # defaults to USA vs Paraguay (2026-06-12, USA home)
  python src/predict.py "Brazil" "Morocco" --neutral --date 2026-06-13
  python src/predict.py --all-wc2026         # predict every upcoming WC2026 fixture -> CSV
"""
import argparse
import numpy as np
import pandas as pd

from config import DATA_PROC, OUTPUTS, FIRST_HALF_FRACTION, DC_TRAIN_FROM
from data import load_results, played_matches, upcoming_matches
from elo import compute_elo, expected_home_winprob
from dixon_coles import DixonColes, _grid_summary
from features import TeamHistory, build_live_features, build_training_table
from team_stats import TeamStats
from stacker import Stacker
from score_engine import ScoreEngine, bivariate_grid
from backtest import build_ht_actuals

OUTCOME_NAMES = ["home", "draw", "away"]
W_ENGINE = 0.5   # weight on the feature-aware ScoreEngine grid vs the calibrated stacker (tuned)


class Predictor:
    def __init__(self, asof=None, rebuild_table=False, verbose=True):
        self.df = load_results()
        self.pm = played_matches(self.df)
        last = self.pm["date"].max()
        self.asof = pd.Timestamp(asof) if asof else (last + pd.Timedelta(days=1))
        if verbose:
            print(f"[predictor] data through {last.date()}  | as-of {self.asof.date()}")
        # Use only matches strictly BEFORE as-of for state (leakage-free retrodiction of past games;
        # for live/future prediction as-of is after the last match so nothing is dropped).
        pm_cut = self.pm[self.pm["date"] < self.asof].reset_index(drop=True)
        self.pm_cut = pm_cut
        # Elo + history (pre-as-of)
        self.pm_elo, self.ratings = compute_elo(pm_cut)
        self.hist = TeamHistory(pm_cut)
        self.stats = TeamStats(self.pm_elo, pm_cut)
        self.ctx = dict(ratings=self.ratings, hist=self.hist, stats=self.stats)
        # serving Dixon-Coles fit on all played data up to as-of
        if verbose:
            print("[predictor] fitting serving Dixon-Coles ...")
        self.dc = DixonColes().fit(pm_cut, ref_date=self.asof, verbose=verbose)
        # stacker on the (cached or rebuilt) training table
        tbl_path = DATA_PROC / "train_table.pkl"
        if rebuild_table or not tbl_path.exists():
            if verbose:
                print("[predictor] building training table ...")
            tbl, _ = build_training_table(self.pm)
            tbl.to_pickle(tbl_path)
        else:
            tbl = pd.read_pickle(tbl_path)
        if verbose:
            print(f"[predictor] fitting stacker on {len(tbl):,} matches ...")
        self.stacker = Stacker().fit(tbl)
        # feature-aware score engine (scorelines + dedicated first-half), fit on the table + HT actuals
        if verbose:
            print("[predictor] fitting feature-aware score engine ...")
        ht = build_ht_actuals()
        self.engine = ScoreEngine().fit(tbl, ht[ht.date < self.asof])

    def predict_match(self, home, away, neutral=False, date=None):
        date = pd.Timestamp(date) if date else self.asof
        feat, dc = build_live_features(home, away, neutral, date, self.ctx, self.dc)
        p_stack = self.stacker.predict_proba_row(feat)  # calibrated 1X2
        # feature-aware score engine -> lambdas -> grids
        lh, la, lhf, laf = self.engine.lambdas(feat)
        grid = bivariate_grid(lh, la, rho=self.dc.rho)
        X, Yi = np.indices(grid.shape)
        grid_p = np.array([grid[X > Yi].sum(), grid[X == Yi].sum(), grid[X < Yi].sum()])
        # final 1X2 = blend(calibrated stacker, feature-aware engine grid)
        p = W_ENGINE * grid_p + (1 - W_ENGINE) * p_stack
        p = p / p.sum()
        # rescale engine grid's outcome regions to the final 1X2 (consistent scorelines)
        adj = grid.copy()
        for region, target in ((X > Yi, p[0]), (X == Yi, p[1]), (X < Yi, p[2])):
            cur = adj[region].sum()
            if cur > 0:
                adj[region] *= target / cur
        adj /= adj.sum()
        ft = _grid_summary(adj, lh, la)
        # half-time from the engine's dedicated first-half model
        hg = bivariate_grid(lhf, laf, rho=self.dc.rho)
        Xh, Yh = np.indices(hg.shape)
        p_hl, p_lv, p_al = float(hg[Xh > Yh].sum()), float(hg[Xh == Yh].sum()), float(hg[Xh < Yh].sum())
        mlh = np.unravel_index(np.argmax(hg), hg.shape)
        leader = "home" if max(p_hl, p_lv, p_al) == p_hl else "level" if max(p_hl, p_lv, p_al) == p_lv else "away"
        fh = dict(p_home_lead=p_hl, p_level=p_lv, p_away_lead=p_al,
                  most_likely_ht_score=(int(mlh[0]), int(mlh[1])),
                  exp_ht_goals_home=float(lhf), exp_ht_goals_away=float(laf), ht_leader=leader)
        rh = self.ratings.get(home, 1500.0)
        ra = self.ratings.get(away, 1500.0)
        outcome = int(np.argmax(p))
        # coherent headline scoreline matching the predicted outcome
        headline_score = {0: ft["best_home_win"][0], 1: ft["best_draw"][0],
                          2: ft["best_away_win"][0]}[outcome]
        # confidence: model is calibrated (ECE~1.1%, optimal T~1.0), so prob == confidence
        top = float(np.sort(p)[-1]); margin = top - float(np.sort(p)[-2])
        ent = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1))) / np.log(3))  # 0..1
        tier = "High" if top >= 0.60 else "Moderate" if top >= 0.45 else "Low (toss-up)"
        return dict(
            home=home, away=away, neutral=neutral, date=date,
            elo_home=rh, elo_away=ra,
            p_home=float(p[0]), p_draw=float(p[1]), p_away=float(p[2]),
            outcome=OUTCOME_NAMES[outcome],
            xg_home=ft["exp_goals_home"], xg_away=ft["exp_goals_away"],
            modal_score=ft["most_likely_score"], headline_score=headline_score,
            top_scorelines=ft["top_scorelines"],
            best_home_win=ft["best_home_win"], best_draw=ft["best_draw"],
            best_away_win=ft["best_away_win"],
            dc_p=(dc["p_home"], dc["p_draw"], dc["p_away"]),
            engine_p=(float(grid_p[0]), float(grid_p[1]), float(grid_p[2])),
            stacker_p=(float(p_stack[0]), float(p_stack[1]), float(p_stack[2])),
            elo_exp_home=expected_home_winprob(rh, ra, neutral),
            confidence=tier, confidence_top=top, margin=margin, entropy=ent,
            ht=fh,
            # full distributions for the market board (over/under, BTTS, score-first, etc.)
            ft_grid=adj, ht_grid=hg, lambdas=(float(lh), float(la), float(lhf), float(laf)),
        )

    def bootstrap_ci(self, home, away, neutral=False, n_boot=40, seed=0):
        """Dixon-Coles parametric uncertainty via bootstrap refits -> 90% CIs on xG and 1X2."""
        win = self.pm[(self.pm.date < self.asof) &
                      (self.pm.date >= pd.Timestamp(DC_TRAIN_FROM))].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        xgh, xga, ph, pdr, pa = [], [], [], [], []
        for _ in range(n_boot):
            samp = win.iloc[rng.integers(0, len(win), len(win))]
            m = DixonColes().fit(samp, ref_date=self.asof)
            pr = m.predict(home, away, neutral=neutral)
            xgh.append(pr["exp_goals_home"]); xga.append(pr["exp_goals_away"])
            ph.append(pr["p_home"]); pdr.append(pr["p_draw"]); pa.append(pr["p_away"])
        q = lambda a: (float(np.percentile(a, 5)), float(np.percentile(a, 95)))
        return dict(n_boot=n_boot, xg_home=q(xgh), xg_away=q(xga),
                    p_home=q(ph), p_draw=q(pdr), p_away=q(pa))

    def pretty(self, r):
        h, a = r["home"], r["away"]
        L = []
        venue = "neutral venue" if r["neutral"] else f"{h} at home"
        L.append("=" * 66)
        L.append(f"  {h}  vs  {a}    ({r['date'].date()}, {venue})")
        L.append("=" * 66)
        L.append(f"  Elo:  {h} {r['elo_home']:.0f}   |   {a} {r['elo_away']:.0f}"
                 f"   (Elo home-exp {r['elo_exp_home']:.2f})")
        L.append("")
        L.append("  FINAL-SCORE PREDICTION")
        L.append(f"    Expected goals (xG):  {h} {r['xg_home']:.2f}  -  {r['xg_away']:.2f}  {a}")
        ms = r["modal_score"]
        L.append(f"    Most likely scoreline (any outcome): {ms[0]}-{ms[1]}")
        hs = r["headline_score"]
        L.append(f"    >>> PREDICTION: {h} {hs[0]} - {hs[1]} {a}   "
                 f"(model pick: {r['outcome'].upper()} win)")
        L.append("    Top scorelines:")
        for (sc, pr) in r["top_scorelines"][:5]:
            L.append(f"        {sc[0]}-{sc[1]}   {pr*100:5.1f}%")
        L.append("")
        L.append("  WIN / DRAW / WIN  (stacked ensemble)")
        L.append(f"    {h:>22s} win : {r['p_home']*100:5.1f}%")
        L.append(f"    {'draw':>22s}     : {r['p_draw']*100:5.1f}%")
        L.append(f"    {a:>22s} win : {r['p_away']*100:5.1f}%")
        L.append(f"    components  DC: {r['dc_p'][0]*100:.0f}/{r['dc_p'][1]*100:.0f}/{r['dc_p'][2]*100:.0f}"
                 f"   engine: {r['engine_p'][0]*100:.0f}/{r['engine_p'][1]*100:.0f}/{r['engine_p'][2]*100:.0f}"
                 f"   stacker: {r['stacker_p'][0]*100:.0f}/{r['stacker_p'][1]*100:.0f}/{r['stacker_p'][2]*100:.0f}")
        L.append(f"    confidence: {r['confidence']}  (top prob {r['confidence_top']*100:.1f}%, "
                 f"margin {r['margin']*100:.1f}pp)  [model calibrated: ECE~1.1%]")
        L.append("")
        fh = r["ht"]
        L.append("  FIRST HALF  (who's winning at the break)")
        L.append(f"    HT xG:  {h} {fh['exp_ht_goals_home']:.2f}  -  {fh['exp_ht_goals_away']:.2f}  {a}")
        L.append(f"    P({h} leads at HT) : {fh['p_home_lead']*100:5.1f}%")
        L.append(f"    P(level at HT)      : {fh['p_level']*100:5.1f}%")
        L.append(f"    P({a} leads at HT) : {fh['p_away_lead']*100:5.1f}%")
        ml = fh["most_likely_ht_score"]
        lead_txt = {"home": f"{h} ahead", "level": "level", "away": f"{a} ahead"}[fh["ht_leader"]]
        L.append(f"    Most likely HT score: {ml[0]}-{ml[1]}   ->  at the break: {lead_txt}")
        L.append("=" * 66)
        return "\n".join(L)


def predict_all_wc2026(pred: Predictor):
    up = upcoming_matches(pred.df)
    wc = up[up["tournament"] == "FIFA World Cup"].copy()
    rows = []
    for _, m in wc.iterrows():
        r = pred.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]),
                               date=m["date"])
        rows.append(dict(
            date=m["date"].date(), home=r["home"], away=r["away"], neutral=r["neutral"],
            pred_outcome=r["outcome"],
            pred_score=f"{r['headline_score'][0]}-{r['headline_score'][1]}",
            modal_score=f"{r['modal_score'][0]}-{r['modal_score'][1]}",
            p_home=round(r["p_home"], 3), p_draw=round(r["p_draw"], 3), p_away=round(r["p_away"], 3),
            xg_home=round(r["xg_home"], 2), xg_away=round(r["xg_away"], 2),
            ht_home_lead=round(r["ht"]["p_home_lead"], 3), ht_level=round(r["ht"]["p_level"], 3),
            ht_away_lead=round(r["ht"]["p_away_lead"], 3),
            ht_pred=r["ht"]["ht_leader"],
        ))
    out = pd.DataFrame(rows)
    path = OUTPUTS / "wc2026_predictions.csv"
    out.to_csv(path, index=False)
    print(f"\nsaved {len(out)} WC2026 predictions -> {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?", default="United States")
    ap.add_argument("away", nargs="?", default="Paraguay")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--date", default="2026-06-12")
    ap.add_argument("--all-wc2026", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--ci", action="store_true", help="add Dixon-Coles bootstrap 90%% CIs")
    args = ap.parse_args()

    pred = Predictor(asof=args.date, rebuild_table=args.rebuild)
    if args.all_wc2026:
        out = predict_all_wc2026(pred)
        print(out.to_string(index=False))
        return
    r = pred.predict_match(args.home, args.away, neutral=args.neutral, date=args.date)
    print("\n" + pred.pretty(r))
    if args.ci:
        print("\n  computing Dixon-Coles bootstrap 90% CIs (40 refits)...")
        ci = pred.bootstrap_ci(args.home, args.away, neutral=args.neutral)
        print(f"    xG {args.home}:  [{ci['xg_home'][0]:.2f}, {ci['xg_home'][1]:.2f}]")
        print(f"    xG {args.away}:  [{ci['xg_away'][0]:.2f}, {ci['xg_away'][1]:.2f}]")
        print(f"    P({args.home} win): [{ci['p_home'][0]*100:.1f}%, {ci['p_home'][1]*100:.1f}%]")
        print(f"    P(draw):       [{ci['p_draw'][0]*100:.1f}%, {ci['p_draw'][1]*100:.1f}%]")
        print(f"    P({args.away} win): [{ci['p_away'][0]*100:.1f}%, {ci['p_away'][1]*100:.1f}%]")


if __name__ == "__main__":
    main()
