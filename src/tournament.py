"""Monte-Carlo simulation of the full 2026 World Cup using the match model.

- 12 real groups (official Dec-2025 draw, data-consistent names).
- Group games sampled from each fixture's ensemble score distribution (DC scoreline grid rescaled so
  its outcome marginals match the stacked 1X2). The two already-played group games use real scores.
- Qualification: top 2 per group + 8 best third-placed (FIFA tiebreakers: pts, GD, GF, then random).
- Knockout: real 2026 Round-of-32 seeding template (winner-vs-third, runner-vs-runner, same-group
  separated; 8 thirds assigned to the 8 winner-slots by constrained matching). Post-R32 tree uses the
  canonical sequential pairing of the official R32 template (the exact FIFA R16+ slotting locks only
  after the group stage). Knockout ties resolved by P(advance) = p_win + p_draw*p_win/(p_win+p_loss).

Outputs each team's probability of reaching each stage and winning the cup.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import OUTPUTS
from features import build_live_features

# Official 2026 group draw (names match results.csv)
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
TEAM_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

# Round-of-32 template: (slotA, slotB). 'Wx'=winner group x, 'Rx'=runner x, ('T',{allowed groups})=third
R32 = [
    ("RA", "RB"), ("WC", "RF"), ("WE", ("T", set("CDFGH"))), ("WF", "RC"),
    ("RE", "RI"), ("WI", ("T", set("CDFGH"))), ("WA", ("T", set("CEFHI"))),
    ("WL", ("T", set("EHIJK"))), ("WG", ("T", set("AEHIJ"))), ("WD", ("T", set("BEFIJ"))),
    ("WH", "RJ"), ("RK", "RL"), ("WB", ("T", set("EFGIJ"))), ("RD", "RG"),
    ("WJ", "RH"), ("WK", ("T", set("DEIJL"))),
]


def _blend_probs(pred, feat, dc):
    """Final 1X2 = blend(calibrated stacker, feature-aware engine grid) — mirrors predict_match."""
    from predict import W_ENGINE
    from score_engine import bivariate_grid
    pstk = np.asarray(pred.stacker.predict_proba_row(feat))
    lh, la, _, _ = pred.engine.lambdas(feat)
    grid = bivariate_grid(lh, la, rho=pred.dc.rho)
    X, Y = np.indices(grid.shape)
    gp = np.array([grid[X > Y].sum(), grid[X == Y].sum(), grid[X < Y].sum()])
    p = W_ENGINE * gp + (1 - W_ENGINE) * pstk
    return p / p.sum(), grid, X, Y


def score_distribution(pred, home, away, neutral):
    """Flattened ensemble score distribution: feature-aware engine grid rescaled so the home/draw/away
    mass matches the blended 1X2. Returns (probs, home_goals, away_goals) flat arrays."""
    feat, dc = build_live_features(home, away, neutral, pred.asof, pred.ctx, pred.dc)
    p, grid, X, Y = _blend_probs(pred, feat, dc)
    grid = grid.copy()
    for region, target in ((X > Y, p[0]), (X == Y, p[1]), (X < Y, p[2])):
        cur = grid[region].sum()
        if cur > 0:
            grid[region] *= target / cur
    grid /= grid.sum()
    return grid.ravel(), X.ravel(), Y.ravel()


class KnockoutCache:
    def __init__(self, pred):
        self.pred = pred
        self.cache = {}

    def p_advance(self, a, b):
        """P(a beats b) in a neutral knockout (draw resolved toward stronger side)."""
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
        # neutral knockout (no home advantage)
        feat, dc = build_live_features(a, b, True, self.pred.asof, self.pred.ctx, self.pred.dc)
        p, _, _, _ = _blend_probs(self.pred, feat, dc)
        pw, pd_, pl = p[0], p[1], p[2]
        denom = pw + pl if (pw + pl) > 0 else 1.0
        pa = pw + pd_ * pw / denom
        self.cache[key] = pa
        self.cache[(b, a)] = 1 - pa
        return pa


def _standings(group_teams, games):
    """games: list of (home, away, hg, ag). Returns teams ranked 1..4 with (pts, gd, gf)."""
    pts = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    ga = {t: 0 for t in group_teams}
    for (h, a, hg, ag) in games:
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    rng = np.random.random(len(group_teams))
    order = sorted(range(len(group_teams)),
                   key=lambda i: (pts[group_teams[i]], pts[group_teams[i]] - 0,
                                  gf[group_teams[i]] - ga[group_teams[i]], gf[group_teams[i]], rng[i]),
                   reverse=True)
    ranked = [group_teams[i] for i in order]
    meta = {t: (pts[t], gf[t] - ga[t], gf[t]) for t in group_teams}
    return ranked, meta


def assign_thirds(third_slots_allowed, qualified_thirds):
    """qualified_thirds: list of (team, group_letter). Returns dict slot_index->team via matching."""
    n = len(qualified_thirds)
    cost = np.full((n, n), 1e6)
    for si, allowed in enumerate(third_slots_allowed):
        for ti, (team, grp) in enumerate(qualified_thirds):
            if grp in allowed:
                cost[si, ti] = 0
    r, c = linear_sum_assignment(cost)
    return {third_slots_allowed_idx[si]: qualified_thirds[ti][0] for si, ti in zip(r, c)}


def simulate(pred, n_sims=10000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    # gather the 72 group fixtures from data
    wc = pred.df[(pred.df.tournament == "FIFA World Cup") & (pred.df.date >= pd.Timestamp("2026-01-01"))]
    fixtures = []
    for _, m in wc.iterrows():
        g = TEAM_GROUP.get(m.home_team)
        if g is None or TEAM_GROUP.get(m.away_team) != g:
            continue
        fx = dict(group=g, home=m.home_team, away=m.away_team, neutral=bool(m.neutral),
                  played=bool(m.played))
        if fx["played"]:
            fx["hg"] = np.full(n_sims, int(m.home_score)); fx["ag"] = np.full(n_sims, int(m.away_score))
        else:
            probs, xg, yg = score_distribution(pred, fx["home"], fx["away"], fx["neutral"])
            idx = rng.choice(len(probs), size=n_sims, p=probs)
            fx["hg"] = xg[idx]; fx["ag"] = yg[idx]
        fixtures.append(fx)
    if verbose:
        print(f"[sim] {len(fixtures)} group fixtures, {n_sims} simulations")

    by_group = {g: [f for f in fixtures if f["group"] == g] for g in GROUPS}
    ko = KnockoutCache(pred)
    teams = [t for ts in GROUPS.values() for t in ts]
    stage_counts = {t: dict(R32=0, R16=0, QF=0, SF=0, F=0, W=0) for t in teams}
    group_win = {t: 0 for t in teams}

    # third-slot bookkeeping
    global third_slots_allowed_idx
    third_slot_positions = [i for i, (sa, sb) in enumerate(R32)
                            if isinstance(sb, tuple)]            # R32 indices that take a third
    third_slots_allowed = [R32[i][1][1] for i in third_slot_positions]
    third_slots_allowed_idx = third_slot_positions

    for s in range(n_sims):
        winners, runners, thirds = {}, {}, []
        for g, fxs in by_group.items():
            games = [(f["home"], f["away"], int(f["hg"][s]), int(f["ag"][s])) for f in fxs]
            ranked, meta = _standings(GROUPS[g], games)
            winners[g] = ranked[0]; runners[g] = ranked[1]
            thirds.append((ranked[2], g, meta[ranked[2]]))
            group_win[ranked[0]] += 1
        # 8 best thirds by (pts, gd, gf)
        thirds_sorted = sorted(thirds, key=lambda x: (x[2][0], x[2][1], x[2][2], rng.random()),
                               reverse=True)
        top8 = [(t, g) for (t, g, m) in thirds_sorted[:8]]
        # all 24 group qualifiers + 8 thirds reach R32
        qualified = set(winners.values()) | set(runners.values()) | {t for t, g in top8}
        for t in qualified:
            stage_counts[t]["R32"] += 1
        third_assign = assign_thirds(third_slots_allowed, top8)

        def resolve_slot(slot):
            if slot == "T":
                return None
            kind, grp = slot[0], slot[1]
            return winners[grp] if kind == "W" else runners[grp]

        # build R32 matchups
        r32_pairs = []
        for i, (sa, sb) in enumerate(R32):
            ta = resolve_slot(sa)
            tb = third_assign[i] if isinstance(sb, tuple) else resolve_slot(sb)
            r32_pairs.append((ta, tb))

        # play knockout rounds (sequential-pairing tree)
        def play_round(pairs, stage_name):
            winners_ = []
            for (a, b) in pairs:
                pa = ko.p_advance(a, b)
                w = a if rng.random() < pa else b
                winners_.append(w)
            for w in winners_:
                stage_counts[w][stage_name] += 1
            return winners_

        r16 = play_round(r32_pairs, "R16")
        r16_pairs = [(r16[i], r16[i + 1]) for i in range(0, 16, 2)]
        qf = play_round(r16_pairs, "QF")
        qf_pairs = [(qf[i], qf[i + 1]) for i in range(0, 8, 2)]
        sf = play_round(qf_pairs, "SF")
        sf_pairs = [(sf[i], sf[i + 1]) for i in range(0, 4, 2)]
        fin = play_round(sf_pairs, "F")
        champ = play_round([(fin[0], fin[1])], "W")[0]

    rows = []
    for t in teams:
        c = stage_counts[t]
        rows.append(dict(team=t, group=TEAM_GROUP[t],
                         win_group=group_win[t] / n_sims,
                         reach_R16=c["R16"] / n_sims, reach_QF=c["QF"] / n_sims,
                         reach_SF=c["SF"] / n_sims, reach_final=c["F"] / n_sims,
                         champion=c["W"] / n_sims, advance_group=c["R32"] / n_sims))
    out = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
    # Monte-Carlo standard error on each probability (binomial): sqrt(p(1-p)/N)
    for c in ["champion", "reach_final", "reach_SF", "advance_group"]:
        out[c + "_se"] = np.sqrt(out[c] * (1 - out[c]) / n_sims)
    return out


if __name__ == "__main__":
    from predict import Predictor
    pred = Predictor(asof="2026-06-13")
    res = simulate(pred, n_sims=20000)
    pd.set_option("display.width", 180)
    print("\n================ WORLD CUP 2026 — title odds (champion % ± 95% Monte-Carlo CI) ================")
    for _, r in res.head(15).iterrows():
        ci = 1.96 * r["champion_se"] * 100
        bar = "#" * int(r["champion"] * 100)
        print(f"  {r['team']:16s} (grp {r['group']})  {r['champion']*100:5.1f}% ± {ci:.1f}   {bar}")
    print(f"\n  >>> MOST LIKELY WINNER: {res.iloc[0]['team']} "
          f"({res.iloc[0]['champion']*100:.1f}% ± {1.96*res.iloc[0]['champion_se']*100:.1f})")
    show = res.copy()
    for c in ["win_group", "advance_group", "reach_QF", "reach_SF", "reach_final", "champion"]:
        show[c] = (show[c] * 100).round(1)
    print("\nUSA:")
    print(show[show.team == "United States"][["team", "group", "win_group", "advance_group",
          "reach_QF", "reach_SF", "reach_final", "champion"]].to_string(index=False))
    res.to_csv(OUTPUTS / "wc2026_tournament_sim.csv", index=False)
    print(f"\nsaved {OUTPUTS/'wc2026_tournament_sim.csv'}")
