"""Novel team-strength / psychology features, computed leakage-free as-of any date.

Built from the Elo-augmented match log (each match knows the pre-match Elo of both sides) plus
goalscorer minutes and shootout records. All queries use only matches strictly before `as_of`.

Features (per team, as-of-date):
  choke_index      : when FAVORED (pre-match Elo edge >= +THRESH), the rate of NOT winning.
                     High = the team drops points against teams it should beat. (the user's idea)
  upset_index      : when UNDERDOG (Elo deficit <= -THRESH), the rate of POSITIVE results (win/draw).
                     High = a giant-killer that punches up.
  bigmatch_ppg     : points-per-game vs strong opponents (opp pre-match Elo >= BIG_ELO).
  elo_momentum     : Elo now minus Elo ~MOM_N matches ago (recent trajectory).
  elo_volatility   : std of the last MOM_N per-match Elo deltas (consistency / streakiness).
  clean_sheet_rate : share of recent matches with 0 conceded.
  fts_rate         : share of recent matches failing to score.
  late_goal_share  : share of the team's scored goals that come in minute >= 76 (late danger).
  late_concede_share: share of conceded goals in minute >= 76 (late fragility).
  shootout_winrate : career penalty-shootout win rate (knockout relevance).
"""
import bisect

import numpy as np
import pandas as pd

FAVORED_THRESH = 40.0    # Elo edge to count as "favored"
BIG_ELO = 1850.0         # opponent Elo to count as a "big match"
MOM_N = 10               # window for momentum / volatility
RECENT_N = 10            # window for clean sheets / fts / choke / upset / bigmatch


class TeamStats:
    def __init__(self, pm_elo: pd.DataFrame, played: pd.DataFrame, goalscorers_csv=None,
                 shootouts_csv=None):
        from config import GOALSCORERS_CSV, SHOOTOUTS_CSV, tournament_tier
        self.team = {}   # team -> dict of sorted lists keyed by date
        # ---- per-match outcome log with pre-match Elo ----
        h = pm_elo["home_team"].values; a = pm_elo["away_team"].values
        hs = pm_elo["home_score"].values; as_ = pm_elo["away_score"].values
        eh = pm_elo["elo_home_pre"].values; ea = pm_elo["elo_away_pre"].values
        we = pm_elo["elo_exp_home"].values
        tourn = pm_elo["tournament"].values
        dates = pm_elo["date"].values
        for i in range(len(pm_elo)):
            mj = tournament_tier(tourn[i]) >= 4
            self._add(h[i], dates[i], eh[i], ea[i], hs[i], as_[i], we[i], mj)
            self._add(a[i], dates[i], ea[i], eh[i], as_[i], hs[i], 1.0 - we[i], mj)
        for t in self.team:
            self.team[t]["dates"] = np.array(self.team[t]["dates"])

        # ---- goal minutes (for/against) per team ----
        g = pd.read_csv(goalscorers_csv or GOALSCORERS_CSV)
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        from data import _parse_minute
        g["m"] = g["minute"].map(_parse_minute)
        g = g.dropna(subset=["m"])
        self.goals_for = {}    # team -> (sorted dates array, minutes array)
        self.goals_against = {}
        gh = g["home_team"].values; ga = g["away_team"].values
        gt = g["team"].values; gm = g["m"].values; gd = g["date"].values
        tmp_for, tmp_against = {}, {}
        for i in range(len(g)):
            scorer_team = gt[i]
            opp = ga[i] if scorer_team == gh[i] else gh[i]
            tmp_for.setdefault(scorer_team, []).append((gd[i], gm[i]))
            tmp_against.setdefault(opp, []).append((gd[i], gm[i]))
        for t, lst in tmp_for.items():
            lst.sort(); self.goals_for[t] = (np.array([x[0] for x in lst]), np.array([x[1] for x in lst]))
        for t, lst in tmp_against.items():
            lst.sort(); self.goals_against[t] = (np.array([x[0] for x in lst]), np.array([x[1] for x in lst]))

        # ---- shootouts ----
        self.so = {}   # team -> (sorted dates, win_bool)
        try:
            s = pd.read_csv(shootouts_csv or SHOOTOUTS_CSV)
            s["date"] = pd.to_datetime(s["date"], errors="coerce")
            tmp = {}
            for _, r in s.iterrows():
                for team in (r["home_team"], r["away_team"]):
                    tmp.setdefault(team, []).append((r["date"], 1 if r["winner"] == team else 0))
            for t, lst in tmp.items():
                lst.sort(); self.so[t] = (np.array([x[0] for x in lst]), np.array([x[1] for x in lst]))
        except Exception:
            pass

    def _add(self, team, date, elo_self, elo_opp, gf, ga, we_self=0.5, is_major=False):
        d = self.team.setdefault(team, dict(dates=[], elo_self=[], elo_opp=[], gf=[], ga=[],
                                            pts=[], won=[], we_resid=[], major=[]))
        d["dates"].append(date)
        d["elo_self"].append(elo_self); d["elo_opp"].append(elo_opp)
        d["gf"].append(gf); d["ga"].append(ga)
        d["won"].append(1 if gf > ga else 0)
        d["pts"].append(3 if gf > ga else 1 if gf == ga else 0)
        w = 1.0 if gf > ga else 0.5 if gf == ga else 0.0
        d["we_resid"].append(w - we_self)        # actual minus Elo-expected result (over/under-perf)
        d["major"].append(1 if is_major else 0)

    def _slice_idx(self, dates_arr, as_of):
        """number of matches strictly before as_of."""
        return bisect.bisect_left(dates_arr, np.datetime64(pd.Timestamp(as_of)))

    def features(self, team, as_of):
        out = dict(choke_index=np.nan, upset_index=np.nan, bigmatch_ppg=np.nan,
                   elo_momentum=0.0, elo_volatility=0.0, clean_sheet_rate=np.nan,
                   fts_rate=np.nan, late_goal_share=np.nan, late_concede_share=np.nan,
                   shootout_winrate=0.5, overperf_vs_elo=0.0, temperament=0.0)
        d = self.team.get(team)
        if d is not None:
            n = self._slice_idx(d["dates"], as_of)
            if n > 0:
                es = np.array(d["elo_self"][:n]); eo = np.array(d["elo_opp"][:n])
                won = np.array(d["won"][:n]); pts = np.array(d["pts"][:n])
                gf = np.array(d["gf"][:n]); ga = np.array(d["ga"][:n])
                resid = np.array(d["we_resid"][:n]); major = np.array(d["major"][:n])
                # choke: favored matches -> non-win rate
                fav = es - eo >= FAVORED_THRESH
                if fav.sum() >= 3:
                    out["choke_index"] = 1.0 - won[fav].mean()
                # upset: underdog matches -> positive-result (pts>=1) rate
                dog = es - eo <= -FAVORED_THRESH
                if dog.sum() >= 3:
                    out["upset_index"] = (pts[dog] >= 1).mean()
                # big-match ppg
                big = eo >= BIG_ELO
                if big.sum() >= 3:
                    out["bigmatch_ppg"] = pts[big].mean()
                rec = slice(max(0, n - MOM_N), n)
                r = slice(max(0, n - RECENT_N), n)
                out["clean_sheet_rate"] = (ga[r] == 0).mean()
                out["fts_rate"] = (gf[r] == 0).mean()
                es_rec = es[rec]
                if len(es_rec) >= 2:
                    out["elo_momentum"] = float(es_rec[-1] - es_rec[0])
                    out["elo_volatility"] = float(np.std(np.diff(es_rec)))
                # NOVEL: over/under-performance vs Elo expectation (recent actual-minus-expected result)
                out["overperf_vs_elo"] = float(resid[r].mean())
                # NOVEL: tournament temperament = major-tournament PPG minus baseline PPG (career)
                if major.sum() >= 3:
                    out["temperament"] = float(pts[major == 1].mean() - pts.mean())
        # late goal shares
        gfa = self.goals_for.get(team)
        if gfa is not None:
            k = self._slice_idx(gfa[0], as_of)
            if k >= 5:
                mins = gfa[1][max(0, k - 40):k]
                out["late_goal_share"] = float((mins >= 76).mean())
        gag = self.goals_against.get(team)
        if gag is not None:
            k = self._slice_idx(gag[0], as_of)
            if k >= 5:
                mins = gag[1][max(0, k - 40):k]
                out["late_concede_share"] = float((mins >= 76).mean())
        # shootouts (career, as-of)
        so = self.so.get(team)
        if so is not None:
            k = self._slice_idx(so[0], as_of)
            if k >= 1:
                out["shootout_winrate"] = float(so[1][:k].mean())
        return out


# overperf_vs_elo and temperament are computed but EXCLUDED from the model: tested, no validated gain
# (signal ceiling). Kept in features() for diagnostics / future use.
NOVEL_COLS = ["choke_index", "upset_index", "bigmatch_ppg", "elo_momentum", "elo_volatility",
              "clean_sheet_rate", "fts_rate", "late_goal_share", "late_concede_share",
              "shootout_winrate"]


if __name__ == "__main__":
    from data import load_results, played_matches
    from elo import compute_elo
    df = load_results(); pm = played_matches(df)
    pm_elo, _ = compute_elo(pm)
    ts = TeamStats(pm_elo, pm)
    for team in ["United States", "Paraguay", "Spain", "Argentina", "Germany"]:
        f = ts.features(team, "2026-06-12")
        print(f"\n{team}:")
        print(f"  choke_index={f['choke_index']:.3f}  upset_index={f['upset_index']:.3f}  "
              f"bigmatch_ppg={f['bigmatch_ppg']:.2f}")
        print(f"  late_goal_share={f['late_goal_share']:.3f}  late_concede_share={f['late_concede_share']:.3f}  "
              f"clean_sheet={f['clean_sheet_rate']:.2f}  shootout={f['shootout_winrate']:.2f}")
        print(f"  elo_momentum={f['elo_momentum']:.1f}  elo_volatility={f['elo_volatility']:.1f}")
