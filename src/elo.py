"""World Football Elo Ratings (eloratings.net spec) computed in-house over all matches.

R_new = R_old + K * G * (W - We)
  W   : 1 win / 0.5 draw / 0 loss (shootouts count as draw via the recorded level score)
  We  : 1 / (10^(-dr/400) + 1), dr = R_home - R_away (+100 home bonus if not neutral)
  K   : match importance (WC=60, qualifiers=40, continental finals=50, friendly=20, ...)
  G   : margin-of-victory multiplier (1, 1.5 for 2, (11+N)/8 for N>=3)

Processes played matches in date order, recording each team's PRE-match rating (leakage-free
features) and exposing the final rating snapshot for predicting future fixtures.
"""
import numpy as np
import pandas as pd

from config import ELO_INIT, ELO_HOME_ADV, elo_k_for_tournament


def _g_multiplier(goal_diff: int) -> float:
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8.0


def compute_elo(played: pd.DataFrame):
    """Returns (played_with_elo, ratings_dict).

    played_with_elo gains columns: elo_home_pre, elo_away_pre, elo_exp_home (We with home adv).
    ratings_dict maps team -> final Elo after the last played match.
    """
    ratings = {}
    n = len(played)
    eh = np.empty(n)
    ea = np.empty(n)
    exp_h = np.empty(n)

    home = played["home_team"].values
    away = played["away_team"].values
    hs = played["home_score"].values
    as_ = played["away_score"].values
    neutral = played["neutral"].values
    tourn = played["tournament"].values

    for i in range(n):
        h, a = home[i], away[i]
        rh = ratings.get(h, ELO_INIT)
        ra = ratings.get(a, ELO_INIT)
        eh[i] = rh
        ea[i] = ra

        adv = 0.0 if neutral[i] else ELO_HOME_ADV
        dr = (rh + adv) - ra
        we = 1.0 / (10 ** (-dr / 400.0) + 1.0)
        exp_h[i] = we

        if hs[i] > as_[i]:
            w = 1.0
        elif hs[i] == as_[i]:
            w = 0.5
        else:
            w = 0.0

        k = elo_k_for_tournament(tourn[i])
        g = _g_multiplier(hs[i] - as_[i])
        delta = k * g * (w - we)
        ratings[h] = rh + delta
        ratings[a] = ra - delta  # zero-sum

    out = played.copy()
    out["elo_home_pre"] = eh
    out["elo_away_pre"] = ea
    out["elo_exp_home"] = exp_h
    return out, ratings


def expected_home_winprob(rh: float, ra: float, neutral: bool) -> float:
    adv = 0.0 if neutral else ELO_HOME_ADV
    dr = (rh + adv) - ra
    return 1.0 / (10 ** (-dr / 400.0) + 1.0)


if __name__ == "__main__":
    from data import load_results, played_matches
    df = load_results()
    pm = played_matches(df)
    pm_elo, ratings = compute_elo(pm)
    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:20]
    print("Top 20 Elo (as of last played match):")
    for t, r in top:
        print(f"  {r:7.1f}  {t}")
    print(f"\nUSA  : {ratings.get('United States'):.1f}")
    print(f"Paraguay: {ratings.get('Paraguay'):.1f}")
    print(f"Elo home win-prob USA vs Paraguay (USA home, non-neutral): "
          f"{expected_home_winprob(ratings['United States'], ratings['Paraguay'], False):.3f}")
