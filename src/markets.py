"""Derive a full betting-market board from a Predictor.predict_match() result.

The model produces a full-time score grid, a first-half score grid, and the per-team goal rates
(lambdas). From those three objects EVERY common pre-match market is just a sum over grid cells or a
closed-form Poisson expression — no extra model needed. For each selection we report:

  prob      : the model's fair probability (0..1)
  fair_c    : fair price in cents  (= prob * 100; on Kalshi a binary contract's price IS a probability)
  entry_c   : the highest price you should pay to still clear a minimum expected return (MIN_ROI).
              EV per $1 contract bought at price c is (prob - c); requiring EV/stake >= MIN_ROI gives
              c <= prob / (1 + MIN_ROI).  So you only "enter" at or below entry_c.

Everything here is pure: no network, no API. It runs entirely from the offline model output.
"""
import math

import numpy as np

MIN_ROI = 0.10   # require >=10% expected return on stake before a price counts as a "good entry"


def _entry_cents(prob, min_roi=MIN_ROI):
    if prob <= 0:
        return None
    c = math.floor((prob / (1.0 + min_roi)) * 100)
    return max(1, c) if c >= 1 else None


def _finalize(rows, min_roi):
    out = []
    for sel, p in rows:
        p = float(max(0.0, min(1.0, p)))
        out.append(dict(sel=sel, prob=p, fair_c=p * 100.0, entry_c=_entry_cents(p, min_roi)))
    return out


def market_board(r, min_roi=MIN_ROI, live=False):
    """r = a Predictor.predict_match() result dict. Returns a list of {title, rows[]} market groups.

    live=True (in-play): r['ft_grid'] is the FINAL-score distribution given the current score+minute,
    and the time-anchored / already-settled markets (first-to-score, half-time, 1st-half) are omitted.
    """
    h, a = r["home"], r["away"]
    g = np.asarray(r["ft_grid"]); hg = np.asarray(r["ht_grid"])
    lh, la, lhf, laf = r["lambdas"]
    ph, pdr, pa = r["p_home"], r["p_draw"], r["p_away"]
    X, Y = np.indices(g.shape)
    total = X + Y
    Xh, Yh = np.indices(hg.shape)
    th = Xh + Yh

    groups = []

    def grp(title, rows):
        groups.append(dict(title=title, rows=_finalize(rows, min_roi)))

    # --- Match result ---
    grp("Match Result (1X2)", [(f"{h} win", ph), ("Draw", pdr), (f"{a} win", pa)])

    # --- Double chance ---
    grp("Double Chance", [(f"{h} or Draw", ph + pdr), (f"{h} or {a}", ph + pa),
                          (f"Draw or {a}", pdr + pa)])

    # --- Draw no bet (refund on draw -> renormalise over win/win) ---
    s = ph + pa
    if s > 0:
        grp("Draw No Bet", [(f"{h}", ph / s), (f"{a}", pa / s)])

    # --- First team to score (two-rate Poisson race over the whole match; pre-match only) ---
    if not live:
        tot = lh + la
        p_none = math.exp(-tot) if tot > 0 else 1.0
        p_h_first = (lh / tot) * (1 - p_none) if tot > 0 else 0.0
        p_a_first = (la / tot) * (1 - p_none) if tot > 0 else 0.0
        grp("First Team to Score", [(f"{h}", p_h_first), (f"{a}", p_a_first), ("No goal", p_none)])

    # --- Total goals over/under ---
    ou = []
    for line in (1.5, 2.5, 3.5):
        over = float(g[total > line].sum())
        ou.append((f"Over {line}", over))
        ou.append((f"Under {line}", 1 - over))
    grp("Total Goals O/U", ou)

    # --- Both teams to score ---
    btts = float(g[(X >= 1) & (Y >= 1)].sum())
    grp("Both Teams To Score", [("Yes", btts), ("No", 1 - btts)])

    # --- Clean sheet ---
    grp("Clean Sheet", [(f"{h} keeps CS", float(g[Y == 0].sum())),
                        (f"{a} keeps CS", float(g[X == 0].sum()))])

    # --- Winning margin ---
    grp("Winning Margin", [
        (f"{h} by 1", float(g[(X - Y) == 1].sum())),
        (f"{h} by 2+", float(g[(X - Y) >= 2].sum())),
        ("Draw", pdr),
        (f"{a} by 1", float(g[(Y - X) == 1].sum())),
        (f"{a} by 2+", float(g[(Y - X) >= 2].sum())),
    ])

    # --- Half-time result (pre-match only; settled once the half is played) ---
    if not live:
        fh = r["ht"]
        grp("Half-Time Result", [(f"{h} lead", fh["p_home_lead"]), ("Level", fh["p_level"]),
                                 (f"{a} lead", fh["p_away_lead"])])

        # --- 1st-half goals ---
        grp("1st-Half Goals O/U", [
            ("Over 0.5", float(hg[th > 0.5].sum())), ("Under 0.5", float(hg[th <= 0.5].sum())),
            ("Over 1.5", float(hg[th > 1.5].sum())), ("Under 1.5", float(hg[th <= 1.5].sum())),
        ])

    # --- Correct score (top 8 from the grid) ---
    flat = [((i, j), float(g[i, j])) for i in range(g.shape[0]) for j in range(g.shape[1])]
    flat.sort(key=lambda kv: -kv[1])
    grp("Correct Score (top 8)", [(f"{i}-{j}", p) for (i, j), p in flat[:8]])

    return groups


def best_values(r, min_roi=MIN_ROI, top=6):
    """Flat list of every selection sorted by model probability — handy for a quick 'what's strong' view."""
    rows = []
    for grpd in market_board(r, min_roi):
        for row in grpd["rows"]:
            rows.append(dict(market=grpd["title"], **row))
    rows.sort(key=lambda d: -d["prob"])
    return rows[:top]
