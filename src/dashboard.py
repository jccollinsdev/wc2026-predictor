"""Live FAIR-VALUE dashboard: the model's odds for every upcoming game, across all common markets,
each with a recommended max entry price (in cents) so you can see a good price to bet at on Kalshi.

100% offline for the core board (no API keys). If an api-football key is configured (src/lineups.py)
the confirmed XI / injuries are shown per game. You place trades manually -- this only tells you what
the model thinks is fair and where the value is.

Usage:
  python src/dashboard.py                  # next 3 days of fixtures -> console + outputs/dashboard.html
  python src/dashboard.py --days 7
  python src/dashboard.py --wc             # ALL upcoming FIFA World Cup fixtures
  python src/dashboard.py --date 2026-06-13
  python src/dashboard.py --roi 0.15       # require >=15% expected return for the 'enter' price
"""
import argparse
import html as _html

import pandas as pd

from config import OUTPUTS
from data import upcoming_matches
from predict import Predictor
from markets import market_board, best_values, MIN_ROI
import lineups as _lineups


def _fixtures(pred, days, wc):
    up = upcoming_matches(pred.df).copy()
    up = up.sort_values("date")
    if wc:
        return up[up["tournament"] == "FIFA World Cup"]
    lo = pred.asof.normalize()
    hi = lo + pd.Timedelta(days=days)
    return up[(up["date"] >= lo) & (up["date"] < hi)]


def _fmt_entry(c):
    return f"{c}¢" if c else "—"


# ----------------------------- console -----------------------------

def console_game(r, board):
    h, a = r["home"], r["away"]
    venue = "neutral" if r["neutral"] else f"{h} home"
    pick = {"home": h, "draw": "Draw", "away": a}[r["outcome"]]
    L = []
    L.append("─" * 74)
    L.append(f"  {r['date'].date()}   {h}  vs  {a}   ({venue})")
    L.append(f"  Elo {r['elo_home']:.0f}-{r['elo_away']:.0f}   |   model pick: {pick.upper()} "
             f"({r['confidence_top']*100:.0f}%)   proj {r['headline_score'][0]}-{r['headline_score'][1]}"
             f"   conf {r['confidence']}")
    note = _lineups.lineup_note(h, a, r["date"])
    if note:
        L.append(f"  lineups: {note}")
    # show a few headline markets compactly
    keep = {"Match Result (1X2)", "First Team to Score", "Total Goals O/U",
            "Both Teams To Score", "Half-Time Result"}
    for grpd in board:
        if grpd["title"] not in keep:
            continue
        L.append(f"  {grpd['title']}")
        for row in grpd["rows"]:
            if grpd["title"] == "Total Goals O/U" and not row["sel"].endswith("2.5"):
                continue
            L.append(f"      {row['sel']:<22s} {row['prob']*100:5.1f}%   "
                     f"fair {row['fair_c']:4.1f}¢   enter ≤ {_fmt_entry(row['entry_c'])}")
    bv = best_values(r, top=4)
    L.append("  highest-conviction picks (model's most confident selections; a true EDGE needs a live price):")
    for d in bv:
        L.append(f"      {d['sel']:<22s} {d['prob']*100:5.1f}%   enter ≤ {_fmt_entry(d['entry_c'])}"
                 f"   [{d['market']}]")
    return "\n".join(L)


# ----------------------------- HTML -----------------------------

_CSS = """
:root{--bg:#0f1419;--card:#1a212b;--card2:#212a36;--line:#2c3744;--txt:#e6edf3;--mut:#8b97a6;
--grn:#2ea043;--grn2:#1f6f33;--amb:#d29922;--red:#c93c37;--accent:#3b82f6;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
header{padding:20px 26px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5;}
header h1{margin:0 0 4px;font-size:20px;}
header .sub{color:var(--mut);font-size:13px;}
.wrap{padding:22px 26px;display:flex;flex-direction:column;gap:22px;max-width:1500px;margin:0 auto;}
.game{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.ghead{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;
gap:10px 18px;align-items:baseline;background:var(--card2);}
.ghead .teams{font-size:17px;font-weight:600;}
.ghead .meta{color:var(--mut);font-size:12.5px;}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;}
.pill.home{background:#13351f;color:#74e08c;} .pill.away{background:#3a1620;color:#f08aa0;}
.pill.draw{background:#33301a;color:#e3c869;}
.conf-High{color:#74e08c;} .conf-Moderate{color:#e3c869;} .conf-Low{color:#f0a08a;}
.lineups{padding:8px 18px;color:#9fb0c0;font-size:12.5px;border-bottom:1px solid var(--line);background:#161d26;}
.boards{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px;padding:16px 18px;}
.mkt{background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:10px 12px;}
.mkt h4{margin:0 0 8px;font-size:13px;color:#cdd9e5;font-weight:600;letter-spacing:.2px;}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
td{padding:3px 4px;vertical-align:middle;}
td.sel{color:var(--txt);} td.num{text-align:right;color:var(--mut);font-variant-numeric:tabular-nums;}
td.entry{text-align:right;font-weight:600;font-variant-numeric:tabular-nums;}
.bar{height:5px;border-radius:3px;background:var(--line);margin-top:3px;overflow:hidden;}
.bar>span{display:block;height:100%;background:linear-gradient(90deg,#3b82f6,#22c55e);}
.legend{color:var(--mut);font-size:12px;padding:0 18px 16px;}
.footer{color:var(--mut);font-size:12px;padding:18px 26px;border-top:1px solid var(--line);}
"""


def _row_html(row):
    pct = row["prob"] * 100
    bar = min(100.0, pct)
    entry = _fmt_entry(row["entry_c"])
    return (f"<tr><td class='sel'>{_html.escape(row['sel'])}"
            f"<div class='bar'><span style='width:{bar:.1f}%'></span></div></td>"
            f"<td class='num'>{pct:.1f}%</td>"
            f"<td class='num'>{row['fair_c']:.1f}¢</td>"
            f"<td class='entry'>≤ {entry}</td></tr>")


def _game_html(r, board):
    h, a = r["home"], r["away"]
    venue = "neutral venue" if r["neutral"] else f"{_html.escape(h)} at home"
    pickcls = r["outcome"]
    pickname = {"home": h, "draw": "Draw", "away": a}[r["outcome"]]
    confcls = "conf-High" if r["confidence"].startswith("High") else \
        "conf-Moderate" if r["confidence"].startswith("Mod") else "conf-Low"
    parts = ["<div class='game'>"]
    parts.append("<div class='ghead'>")
    parts.append(f"<span class='teams'>{_html.escape(h)} <span style='color:#8b97a6'>vs</span> {_html.escape(a)}</span>")
    parts.append(f"<span class='pill {pickcls}'>{_html.escape(pickname)} {r['confidence_top']*100:.0f}%</span>")
    parts.append(f"<span class='meta'>{r['date'].date()} &middot; {venue} &middot; "
                 f"Elo {r['elo_home']:.0f}–{r['elo_away']:.0f} &middot; "
                 f"proj {r['headline_score'][0]}–{r['headline_score'][1]} &middot; "
                 f"<span class='{confcls}'>{_html.escape(r['confidence'])}</span></span>")
    parts.append("</div>")
    note = _lineups.lineup_note(h, a, r["date"])
    if note:
        parts.append(f"<div class='lineups'>\U0001f9e9 {_html.escape(note)}</div>")
    parts.append("<div class='boards'>")
    for grpd in board:
        parts.append("<div class='mkt'><h4>" + _html.escape(grpd["title"]) + "</h4><table>")
        parts.append("".join(_row_html(row) for row in grpd["rows"]))
        parts.append("</table></div>")
    parts.append("</div></div>")
    return "".join(parts)


def build_html(results_boards, asof, min_roi):
    body = [f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>WC2026 Fair-Value Dashboard</title><style>{_CSS}</style></head><body>"]
    body.append("<header><h1>⚽ WC2026 — Model Fair-Value Dashboard</h1>"
                f"<div class='sub'>As of {asof.date()} &middot; "
                f"“Fair” = model's true probability in cents &middot; "
                f"“Enter ≤” = max price to still clear "
                f"{min_roi*100:.0f}% expected return &middot; "
                f"lineups: {'ON' if _lineups.available() else 'add API_FOOTBALL_KEY to enable'}</div></header>")
    body.append("<div class='wrap'>")
    for r, board in results_boards:
        body.append(_game_html(r, board))
    body.append("</div>")
    body.append("<div class='footer'>Model: Dixon–Coles + Elo + XGBoost stacker + feature-aware "
                "score engine. Prices are model fair value, not a guarantee. You place all trades "
                "manually. On Kalshi a contract's price in cents already equals its implied probability, "
                "so buy YES only at or below the “Enter ≤” price.</div>")
    body.append("</body></html>")
    return "".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="fixtures within the next N days (default 3)")
    ap.add_argument("--wc", action="store_true", help="all upcoming FIFA World Cup fixtures instead")
    ap.add_argument("--date", default=None, help="as-of date (defaults to day after last result)")
    ap.add_argument("--roi", type=float, default=MIN_ROI, help="required expected ROI for entry price")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    pred = Predictor(asof=args.date, rebuild_table=args.rebuild)
    fx = _fixtures(pred, args.days, args.wc)
    if len(fx) == 0:
        print("no upcoming fixtures in the selected window.")
        return
    print(f"\n[dashboard] {len(fx)} fixture(s) | entry requires >= {args.roi*100:.0f}% expected ROI\n")

    results_boards = []
    for _, m in fx.iterrows():
        r = pred.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]),
                               date=m["date"])
        board = market_board(r, min_roi=args.roi)
        results_boards.append((r, board))
        print(console_game(r, board))
    print("─" * 74)

    if not args.no_html:
        out = OUTPUTS / "dashboard.html"
        out.write_text(build_html(results_boards, pred.asof, args.roi))
        print(f"\nsaved interactive dashboard -> {out}")
        print(f"open it with:  open '{out}'")


if __name__ == "__main__":
    main()
