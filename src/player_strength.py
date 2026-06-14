"""National-team squad strength from EA Sports FC 26 player ratings (no-auth GitHub mirror).

Rather than fuzzy-matching announced lineups (fragile), we aggregate each nation's best-rated players
by `nationality_name` — a robust, current squad-strength prior. This is exactly the signal Elo misses:
e.g. the USA's players rate well above their Elo (big-league talent vs a weak-confederation rating).

Per team: mean of top-23 overall (depth), mean of top-11 (first-choice strength), attack/defense
sub-ratings (top players by position group), and best-goalkeeper rating.

Caveat: ratings are 2026-current and applied as a static team prior to all matches (squad strength is
slow-moving; this is standard practice with FIFA/market-value covariates). Exactly right for WC2026.
"""
import functools
import unicodedata

import pandas as pd

from config import DATA_RAW

PLAYERS_CSV = DATA_RAW / "players.csv"               # EA FC 26 (current)
MULTIYEAR_CSV = DATA_RAW / "players_multiyear.csv"   # FIFA 15-24 stacked (fifa_version col)

# data team name -> EA FC nationality_name (only the ones that don't match exactly)
NATION_MAP = {
    "South Korea": "Korea Republic",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye",
    "Curaçao": "Curacao",
    "Republic of Ireland": "Republic of Ireland",
    "China PR": "China PR",
}

ATT_POS = {"ST", "CF", "LW", "RW", "LM", "RM", "CAM"}
DEF_POS = {"CB", "LB", "RB", "LWB", "RWB", "CDM"}

PLAYER_COLS = ["squad_ovr", "squad_top11", "squad_att", "squad_def", "squad_gk"]


def _norm(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().replace(".", "").strip()


def _aggregate(p):
    """Build {nationality -> strength dict} from a player dataframe."""
    p = p.dropna(subset=["overall", "nationality_name"]).copy()
    p["primary_pos"] = p["player_positions"].astype(str).str.split(",").str[0].str.strip()
    out = {}
    for nat, g in p.groupby("nationality_name"):
        ovr = g["overall"].sort_values(ascending=False)
        top23 = ovr.head(23).mean(); top11 = ovr.head(11).mean()
        att = g[g.primary_pos.isin(ATT_POS)]["overall"].sort_values(ascending=False).head(4).mean()
        dfn = g[g.primary_pos.isin(DEF_POS)]["overall"].sort_values(ascending=False).head(4).mean()
        gk = g[g.primary_pos == "GK"]["overall"].max()
        out[nat] = dict(squad_ovr=float(top23), squad_top11=float(top11),
                        squad_att=float(att) if att == att else float(top11),
                        squad_def=float(dfn) if dfn == dfn else float(top11),
                        squad_gk=float(gk) if gk == gk else float(top11))
    return out


def _drop_injured(p):
    """Remove ruled-out players from the FC26 pool (match by normalized name within nationality)."""
    from injuries import injured_by_nation
    inj = injured_by_nation()
    if not inj:
        return p
    keep = pd.Series(True, index=p.index)
    name_norm = (p["short_name"].astype(str) + " " + p["long_name"].astype(str)).map(_norm)
    nat = p["nationality_name"].astype(str)
    for nation, players in inj.items():
        for pl in players:
            pln = _norm(pl)
            hit = (nat == nation) & name_norm.str.contains(pln, regex=False)
            keep &= ~hit
    return p[keep]


@functools.lru_cache(maxsize=1)
def _editions():
    """{fifa_version(int) -> {nationality -> strength}} for editions 15-24 (multiyear) + 26 (FC26).
    Key 'inj' holds the FC26 table with ruled-out players removed (for live injury-adjusted predictions)."""
    ed = {}
    my = pd.read_csv(MULTIYEAR_CSV, low_memory=False)
    for v, g in my.groupby("fifa_version"):
        ed[int(v)] = _aggregate(g)
    fc26 = pd.read_csv(PLAYERS_CSV, low_memory=False)
    ed[26] = _aggregate(fc26)
    ed["inj"] = _aggregate(_drop_injured(fc26))
    return ed


def _edition_for_year(year):
    ed = _editions()
    avail = sorted(k for k in ed.keys() if isinstance(k, int))   # [15..24, 26]
    if year is None:
        return max(avail)                     # latest (26)
    v = int(year) - 2000
    le = [a for a in avail if a <= v]
    return le[-1] if le else avail[0]         # largest edition <= year, else earliest


def team_strength(team, year=None, injured=False):
    ed = _editions()
    key = "inj" if (injured and _edition_for_year(year) == 26) else _edition_for_year(year)
    tbl = ed[key]
    norm_index = {_norm(n): n for n in tbl}
    if team in NATION_MAP and NATION_MAP[team] in tbl:
        return tbl[NATION_MAP[team]]
    if team in tbl:
        return tbl[team]
    if _norm(team) in norm_index:
        return tbl[norm_index[_norm(team)]]
    return {c: float("nan") for c in PLAYER_COLS}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from tournament import GROUPS
    wc = sorted({t for ts in GROUPS.values() for t in ts})
    rows = [dict(team=t, **team_strength(t)) for t in wc]
    d = pd.DataFrame(rows).sort_values("squad_ovr", ascending=False)
    miss = d[d.squad_ovr.isna()]
    print(f"WC teams with squad data: {d.squad_ovr.notna().sum()}/48  (missing: {list(miss.team)})")
    print("\nTop 10 squads by depth (top-23 overall):")
    print(d.head(10).round(1).to_string(index=False))
    print("\nUSA vs Paraguay squad strength:")
    print(d[d.team.isin(["United States", "Paraguay"])].round(1).to_string(index=False))
