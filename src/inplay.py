"""In-play (live) re-pricing: given a match's pre-kickoff expected goals (lambdas) plus the CURRENT
score and minute, recompute the full-time score distribution and every market.

Model: goals arrive as Poisson processes. The remaining expected goals for each team scale with the
fraction of the match still to play:

    lambda_remaining = lambda_full * (T_end - minute) / 90        (clamped at 0)

The final score = current score + remaining goals (two independent Poissons in-play). So at 1-0 in
the 88th minute the remaining lambdas are tiny, P(another goal) is a few percent, and "home to win by
2+" collapses toward zero -- and keeps shrinking every minute. That is exactly the live behaviour you
want, and it falls straight out of the Poisson clock.

We use independent Poissons for the *remaining* goals (the Dixon-Coles low-score correction is a
kickoff-time dependence; once the game is running the remaining goals are well modelled as independent).
"""
import math

import numpy as np

from markets import market_board

T_END_DEFAULT = 94.0   # nominal final whistle incl. typical stoppage (override per-half if known)


def remaining_lambdas(lh_full, la_full, minute, t_end=T_END_DEFAULT):
    frac = max(0.0, (t_end - float(minute)) / 90.0)
    return lh_full * frac, la_full * frac


def _pois_col(lmbda, kmax):
    k = np.arange(kmax + 1)
    lmbda = max(float(lmbda), 1e-12)
    logp = -lmbda + k * np.log(lmbda) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def final_grid(lh_full, la_full, minute, score_h, score_a, t_end=T_END_DEFAULT, kmax=8):
    """Final-score grid given current state. Rows = final home goals, cols = final away goals."""
    lhr, lar = remaining_lambdas(lh_full, la_full, minute, t_end)
    rem = np.outer(_pois_col(lhr, kmax), _pois_col(lar, kmax))
    rem /= rem.sum()
    H, A = score_h + kmax + 1, score_a + kmax + 1
    g = np.zeros((H, A))
    g[score_h:score_h + kmax + 1, score_a:score_a + kmax + 1] = rem
    return g, lhr, lar


def inplay_board(home, away, lh_full, la_full, minute, score_h, score_a, neutral=False,
                 t_end=T_END_DEFAULT, min_roi=0.10):
    """Return (state_dict, market_board) for a live game state."""
    g, lhr, lar = final_grid(lh_full, la_full, minute, score_h, score_a, t_end)
    X, Y = np.indices(g.shape)
    ph, pdr, pa = float(g[X > Y].sum()), float(g[X == Y].sum()), float(g[X < Y].sum())
    # most-likely FINAL score
    mh, ma = np.unravel_index(int(np.argmax(g)), g.shape)
    r = dict(
        home=home, away=away, neutral=neutral,
        p_home=ph, p_draw=pdr, p_away=pa,
        ft_grid=g, ht_grid=g, lambdas=(lhr, lar, 0.0, 0.0),
        ht=dict(p_home_lead=0.0, p_level=0.0, p_away_lead=0.0),
        score=(int(score_h), int(score_a)), minute=float(minute),
        proj_final=(int(mh), int(ma)),
        exp_final_home=score_h + lhr, exp_final_away=score_a + lar,
        outcome=("home" if ph >= pdr and ph >= pa else "draw" if pdr >= pa else "away"),
        confidence_top=max(ph, pdr, pa),
    )
    board = market_board(r, min_roi=min_roi, live=True)
    return r, board


if __name__ == "__main__":
    # demonstrate the decay: home pre-match xG 1.6, away 1.0, currently 1-0, as the clock runs.
    print("Home pre-match xG 1.6, Away 1.0 | current score 1-0 | P(home win) & P(home by 2+) vs minute")
    for minute in (0, 30, 60, 75, 85, 88, 90, 93):
        r, _ = inplay_board("Home", "Away", 1.6, 1.0, minute, 1, 0)
        g = r["ft_grid"]; X, Y = np.indices(g.shape)
        by2 = float(g[(X - Y) >= 2].sum())
        print(f"  min {minute:>2}:  P(home win) {r['p_home']*100:5.1f}%   "
              f"P(draw) {r['p_draw']*100:4.1f}%   P(home by 2+) {by2*100:5.1f}%   "
              f"proj final {r['proj_final'][0]}-{r['proj_final'][1]}")
