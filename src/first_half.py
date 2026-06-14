"""First-half prediction via Poisson thinning of the Dixon-Coles intensities.

A Poisson process thinned by fraction f is Poisson(f*lambda). With f = first-half goal fraction
(measured 0.4474 from goalscorers.csv), first-half intensities are f*lambda_home, f*lambda_away.
Build a half-time scoreline grid (re-applying the DC tau low-score correction, since HT is even
more draw/low-score dominated) and read off who leads at half-time.
"""
import numpy as np

from config import FIRST_HALF_FRACTION, DC_MAX_GOALS


def _log_fact(kmax):
    import math
    return np.array([math.lgamma(k + 1) for k in range(kmax + 1)])


def ht_probs_from_lambdas(lh_full, la_full, f=FIRST_HALF_FRACTION, rho=-0.07, kmax=8):
    """Half-time lead probabilities from full-match lambdas via Poisson thinning. Standalone
    (no model instance) for backtest use against actual minute<=45 goals."""
    lh, la = f * lh_full, f * la_full
    k = np.arange(kmax + 1)
    lf = _log_fact(kmax)
    ph = np.exp(-lh + k * np.log(lh) - lf)
    pa = np.exp(-la + k * np.log(la) - lf)
    grid = np.outer(ph, pa)
    grid[0, 0] *= (1.0 - lh * la * rho)
    grid[0, 1] *= (1.0 + lh * rho)
    grid[1, 0] *= (1.0 + la * rho)
    grid[1, 1] *= (1.0 - rho)
    grid = np.clip(grid, 0, None)
    grid /= grid.sum()
    idx = np.indices(grid.shape)
    X, Y = idx[0], idx[1]
    return float(grid[X > Y].sum()), float(grid[X == Y].sum()), float(grid[X < Y].sum())


def predict_first_half(model, home_team, away_team, neutral=False,
                       f=FIRST_HALF_FRACTION, kmax=DC_MAX_GOALS):
    lh, la, missing = model._lambdas(home_team, away_team, neutral)
    lh1, la1 = f * lh, f * la
    grid = model.score_grid(lh1, la1, kmax=kmax, apply_tau=True, rho=model.rho)
    idx = np.indices(grid.shape)
    X, Y = idx[0], idx[1]
    p_home_lead = float(grid[X > Y].sum())
    p_level = float(grid[X == Y].sum())
    p_away_lead = float(grid[X < Y].sum())
    ml = np.unravel_index(np.argmax(grid), grid.shape)
    leader = ("home" if p_home_lead == max(p_home_lead, p_level, p_away_lead)
              else "level" if p_level == max(p_home_lead, p_level, p_away_lead)
              else "away")
    return dict(
        f=f,
        ht_lambda_home=lh1, ht_lambda_away=la1,
        p_home_lead=p_home_lead, p_level=p_level, p_away_lead=p_away_lead,
        most_likely_ht_score=(int(ml[0]), int(ml[1])),
        exp_ht_goals_home=float(lh1), exp_ht_goals_away=float(la1),
        ht_leader=leader,
    )


if __name__ == "__main__":
    import pandas as pd
    from data import load_results, played_matches
    from dixon_coles import DixonColes

    df = load_results()
    pm = played_matches(df)
    model = DixonColes().fit(pm, ref_date="2026-06-12")
    fh = predict_first_half(model, "United States", "Paraguay", neutral=False)
    print("USA vs Paraguay first half:")
    print(f"  HT xG: USA {fh['exp_ht_goals_home']:.2f} - {fh['exp_ht_goals_away']:.2f} Paraguay")
    print(f"  P(USA leads HT)={fh['p_home_lead']:.3f}  P(level)={fh['p_level']:.3f}  "
          f"P(Paraguay leads HT)={fh['p_away_lead']:.3f}")
    print(f"  most likely HT score: {fh['most_likely_ht_score']}  -> leader: {fh['ht_leader']}")

    # validation: average HT split across many historical fixtures should match ~44/28/16 lit.
    import numpy as np
    sample = pm[(pm.date >= "2022-01-01") & (pm.date < "2026-06-12")].sample(300, random_state=1)
    hl, lv, al = [], [], []
    for _, r in sample.iterrows():
        fh = predict_first_half(model, r.home_team, r.away_team, neutral=bool(r.neutral))
        hl.append(fh["p_home_lead"]); lv.append(fh["p_level"]); al.append(fh["p_away_lead"])
    print(f"\nModel-implied avg HT split over 300 fixtures: "
          f"home-lead {np.mean(hl):.3f} / level {np.mean(lv):.3f} / away-lead {np.mean(al):.3f}")
    print("(literature reference ~ 0.28 / 0.44 / 0.16 for league home/away; internationals vary)")
