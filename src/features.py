"""Leakage-free feature engineering for the 1X2 stacker.

All features for a match use ONLY information available before kickoff:
  - Elo: pre-match ratings (recorded during the single forward Elo pass)
  - Rolling form (last-N): goals for/against, points, goal diff, computed from prior matches
  - Rest days since each team's previous match
  - Recent head-to-head goal difference
  - Dixon-Coles implied probs/lambdas via WALK-FORWARD fits (model fit at the cutoff <= match date)

Used both to build the training table and to construct features for a single live fixture.
"""
import numpy as np
import pandas as pd

from config import FORM_WINDOW, tournament_tier
from elo import compute_elo
from dixon_coles import DixonColes
from team_stats import TeamStats, NOVEL_COLS
from player_strength import team_strength, PLAYER_COLS
from confederation import conf_code
from xg_ratings import xg_features, XG_COLS


class TeamHistory:
    """Per-team ordered match log for fast 'as-of-date' form/rest/H2H queries."""

    def __init__(self, played: pd.DataFrame):
        self.by_team = {}          # team -> list of (date, gf, ga, points)
        self.meetings = {}         # frozenset({A,B}) -> list of (date, home, hs, as_)
        home = played["home_team"].values
        away = played["away_team"].values
        hs = played["home_score"].values
        as_ = played["away_score"].values
        dates = played["date"].values
        for i in range(len(played)):
            h, a = home[i], away[i]
            self.by_team.setdefault(h, []).append((dates[i], hs[i], as_[i],
                                                    3 if hs[i] > as_[i] else 1 if hs[i] == as_[i] else 0))
            self.by_team.setdefault(a, []).append((dates[i], as_[i], hs[i],
                                                    3 if as_[i] > hs[i] else 1 if as_[i] == hs[i] else 0))
            key = frozenset((h, a))
            self.meetings.setdefault(key, []).append((dates[i], h, hs[i], as_[i]))

    def form(self, team, as_of, n=FORM_WINDOW):
        log = self.by_team.get(team)
        if not log:
            return dict(gf=np.nan, ga=np.nan, pts=np.nan, gd=np.nan, rest=np.nan, nmatch=0)
        as_of = np.datetime64(as_of)
        prior = [r for r in log if r[0] < as_of]
        if not prior:
            return dict(gf=np.nan, ga=np.nan, pts=np.nan, gd=np.nan, rest=np.nan, nmatch=0)
        last = prior[-n:]
        gf = np.mean([r[1] for r in last])
        ga = np.mean([r[2] for r in last])
        pts = np.mean([r[3] for r in last])
        rest = (as_of - prior[-1][0]) / np.timedelta64(1, "D")
        return dict(gf=gf, ga=ga, pts=pts, gd=gf - ga, rest=min(rest, 400.0), nmatch=len(prior))

    def h2h_gd(self, home_team, away_team, as_of, k=8):
        """Recent meetings avg goal diff from the perspective of `home_team`."""
        key = frozenset((home_team, away_team))
        ms = self.meetings.get(key)
        if not ms:
            return 0.0
        as_of = np.datetime64(as_of)
        prior = [m for m in ms if m[0] < as_of][-k:]
        if not prior:
            return 0.0
        diffs = []
        for (_, h, hsc, asc) in prior:
            diffs.append((hsc - asc) if h == home_team else (asc - hsc))
        return float(np.mean(diffs))


def _dc_walkforward_models(played, start_year=2012, end_year=2027,
                           dc_time_decay=None, dc_friendly_weight=None):
    """Fit one DC model per yearly cutoff (ref_date = July 1 each year). Returns sorted list."""
    from config import DC_TIME_DECAY, DC_FRIENDLY_WEIGHT
    xi = DC_TIME_DECAY if dc_time_decay is None else dc_time_decay
    fw = DC_FRIENDLY_WEIGHT if dc_friendly_weight is None else dc_friendly_weight
    cutoffs = [pd.Timestamp(f"{y}-07-01") for y in range(start_year, end_year + 1)]
    models = []
    for c in cutoffs:
        try:
            m = DixonColes(time_decay=xi, friendly_weight=fw).fit(played, ref_date=c)
            models.append((c, m))
        except ValueError:
            pass
    return models


def _pick_model(models, date):
    chosen = None
    for c, m in models:
        if c <= date:
            chosen = m
        else:
            break
    return chosen if chosen is not None else models[0][1]


FEATURE_COLS = [
    "elo_diff", "elo_exp_home", "is_neutral",
    "dc_p_home", "dc_p_draw", "dc_p_away", "dc_lambda_home", "dc_lambda_away",
    "form_gd_home", "form_gd_away", "form_pts_home", "form_pts_away",
    "form_gf_home", "form_ga_home", "form_gf_away", "form_ga_away",
    "rest_home", "rest_away", "rest_diff", "tier", "h2h_gd",
] + [f"home_{c}" for c in NOVEL_COLS] + [f"away_{c}" for c in NOVEL_COLS] \
  + [f"home_{c}" for c in PLAYER_COLS] + [f"away_{c}" for c in PLAYER_COLS] \
  + ["squad_ovr_diff", "squad_att_vs_def_diff",
     "elo_home_pre", "elo_away_pre", "home_conf", "away_conf", "cross_conf"]
# (xG features were tested in the stacker too — no validated gain — so excluded.)


def build_training_table(played: pd.DataFrame, train_from_year=2012, verbose=True,
                         dc_time_decay=None, dc_friendly_weight=None):
    pm_elo, ratings = compute_elo(played)
    hist = TeamHistory(played)
    stats = TeamStats(pm_elo, played)
    if verbose:
        print("fitting walk-forward DC models...")
    models = _dc_walkforward_models(played, dc_time_decay=dc_time_decay,
                                    dc_friendly_weight=dc_friendly_weight)
    if verbose:
        print(f"  {len(models)} DC cutoff models fit")

    rows = []
    sub = pm_elo[pm_elo["date"] >= pd.Timestamp(f"{train_from_year}-01-01")]
    for _, r in sub.iterrows():
        dcm = _pick_model(models, r["date"])
        dc = dcm.predict(r["home_team"], r["away_team"], neutral=bool(r["neutral"]))
        fh = hist.form(r["home_team"], r["date"])
        fa = hist.form(r["away_team"], r["date"])
        sh = stats.features(r["home_team"], r["date"])
        sa = stats.features(r["away_team"], r["date"])
        novel = {f"home_{c}": sh[c] for c in NOVEL_COLS}
        novel.update({f"away_{c}": sa[c] for c in NOVEL_COLS})
        _yr = r["date"].year
        ph = team_strength(r["home_team"], _yr); pa = team_strength(r["away_team"], _yr)
        novel.update({f"home_{c}": ph[c] for c in PLAYER_COLS})
        novel.update({f"away_{c}": pa[c] for c in PLAYER_COLS})
        novel["squad_ovr_diff"] = ph["squad_ovr"] - pa["squad_ovr"]
        novel["squad_att_vs_def_diff"] = (ph["squad_att"] - pa["squad_def"]) - (pa["squad_att"] - ph["squad_def"])
        novel["elo_home_pre"] = r["elo_home_pre"]; novel["elo_away_pre"] = r["elo_away_pre"]
        ch, ca = conf_code(r["home_team"]), conf_code(r["away_team"])
        novel["home_conf"] = ch; novel["away_conf"] = ca; novel["cross_conf"] = int(ch != ca)
        rows.append(dict(
            date=r["date"], home_team=r["home_team"], away_team=r["away_team"], result=r["result"],
            home_score=r["home_score"], away_score=r["away_score"],
            dc_ml_home=dc["most_likely_score"][0], dc_ml_away=dc["most_likely_score"][1],
            elo_diff=r["elo_home_pre"] - r["elo_away_pre"], elo_exp_home=r["elo_exp_home"],
            is_neutral=int(r["neutral"]),
            dc_p_home=dc["p_home"], dc_p_draw=dc["p_draw"], dc_p_away=dc["p_away"],
            dc_lambda_home=dc["lambda_home"], dc_lambda_away=dc["lambda_away"],
            form_gd_home=fh["gd"], form_gd_away=fa["gd"],
            form_pts_home=fh["pts"], form_pts_away=fa["pts"],
            form_gf_home=fh["gf"], form_ga_home=fh["ga"],
            form_gf_away=fa["gf"], form_ga_away=fa["ga"],
            rest_home=fh["rest"], rest_away=fa["rest"],
            rest_diff=(fh["rest"] - fa["rest"]) if (fh["rest"] == fh["rest"] and fa["rest"] == fa["rest"]) else np.nan,
            tier=tournament_tier(r["tournament"]), h2h_gd=hist.h2h_gd(r["home_team"], r["away_team"], r["date"]),
            **novel,
        ))
    tbl = pd.DataFrame(rows)
    return tbl, dict(ratings=ratings, hist=hist, models=models, pm_elo=pm_elo)


def build_live_features(home_team, away_team, neutral, date, ctx, dc_model):
    """Feature row (dict) for a single upcoming fixture, using context built from played data."""
    ratings, hist = ctx["ratings"], ctx["hist"]
    from elo import expected_home_winprob
    rh = ratings.get(home_team, 1500.0)
    ra = ratings.get(away_team, 1500.0)
    dc = dc_model.predict(home_team, away_team, neutral=neutral)
    fh = hist.form(home_team, date)
    fa = hist.form(away_team, date)
    stats = ctx["stats"]
    sh = stats.features(home_team, date)
    sa = stats.features(away_team, date)
    novel = {f"home_{c}": sh[c] for c in NOVEL_COLS}
    novel.update({f"away_{c}": sa[c] for c in NOVEL_COLS})
    _yr = pd.Timestamp(date).year
    ph = team_strength(home_team, _yr, injured=True); pa = team_strength(away_team, _yr, injured=True)
    novel.update({f"home_{c}": ph[c] for c in PLAYER_COLS})
    novel.update({f"away_{c}": pa[c] for c in PLAYER_COLS})
    novel["squad_ovr_diff"] = ph["squad_ovr"] - pa["squad_ovr"]
    novel["squad_att_vs_def_diff"] = (ph["squad_att"] - pa["squad_def"]) - (pa["squad_att"] - ph["squad_def"])
    novel["elo_home_pre"] = rh; novel["elo_away_pre"] = ra
    ch, ca = conf_code(home_team), conf_code(away_team)
    novel["home_conf"] = ch; novel["away_conf"] = ca; novel["cross_conf"] = int(ch != ca)
    feat = dict(
        elo_diff=rh - ra, elo_exp_home=expected_home_winprob(rh, ra, neutral),
        is_neutral=int(neutral),
        dc_p_home=dc["p_home"], dc_p_draw=dc["p_draw"], dc_p_away=dc["p_away"],
        dc_lambda_home=dc["lambda_home"], dc_lambda_away=dc["lambda_away"],
        form_gd_home=fh["gd"], form_gd_away=fa["gd"],
        form_pts_home=fh["pts"], form_pts_away=fa["pts"],
        form_gf_home=fh["gf"], form_ga_home=fh["ga"],
        form_gf_away=fa["gf"], form_ga_away=fa["ga"],
        rest_home=fh["rest"], rest_away=fa["rest"],
        rest_diff=(fh["rest"] - fa["rest"]) if (fh["rest"] == fh["rest"] and fa["rest"] == fa["rest"]) else np.nan,
        tier=4, h2h_gd=hist.h2h_gd(home_team, away_team, date),
        **novel,
    )
    return feat, dc


if __name__ == "__main__":
    from data import load_results, played_matches
    df = load_results()
    pm = played_matches(df)
    tbl, ctx = build_training_table(pm, train_from_year=2012)
    print(f"\ntraining table: {len(tbl):,} rows x {len(FEATURE_COLS)} features")
    print("result distribution:", tbl["result"].value_counts(normalize=True).round(3).to_dict())
    print(tbl[["date", "home_team", "away_team", "result", "elo_diff",
               "dc_p_home", "dc_p_away", "form_gd_home", "rest_home"]].tail(6).to_string(index=False))
    from config import DATA_PROC
    tbl.to_pickle(DATA_PROC / "train_table.pkl")
    print(f"saved {DATA_PROC / 'train_table.pkl'}")
