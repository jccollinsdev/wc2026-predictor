"""Localhost live fair-value server.

  python src/server.py                 # http://127.0.0.1:8000  (next 3 days of fixtures)
  python src/server.py --port 8000 --days 7

What it does, refreshing every 60 s:
  * PRE-MATCH board for every upcoming game (model fair odds + recommended entry price per market)
  * LIVE in-play board for any game in progress -- re-priced from the CURRENT score + minute, so the
    odds move as the match unfolds (e.g. at 1-0 in the 88th minute, "win by 2+" keeps shrinking).
  * A manual "what-if" in-play tool you can use right now with NO API key: type a score + minute and
    watch every market re-price.

Live games auto-populate ONLY when API_FOOTBALL_KEY is set (api-football). Everything else is offline.
You place all trades yourself; this just shows the model's fair value and a good entry price.
"""
import argparse
import html as _html

import pandas as pd
from flask import Flask, request

from config import OUTPUTS
from data import upcoming_matches
from predict import Predictor
from markets import market_board, MIN_ROI
from inplay import inplay_board
from dashboard import _CSS, _row_html, _game_html
import lineups as _lineups
import kalshi as _kalshi
import edges as _edges
import paper as _paper
import sharp as _sharp
import time
import urllib.parse

app = Flask(__name__)
STATE = {}            # filled by build_state() at startup
_LAMBDA = {}          # (home, away, neutral) -> (lh_full, la_full)

# light alias map: api-football team name -> our model name (extend as needed)
ALIAS = {
    "USA": "United States", "South Korea": "Korea Republic", "Ivory Coast": "Côte d'Ivoire",
    "Turkey": "Türkiye", "Czech Republic": "Czechia", "Cape Verde": "Cabo Verde",
    "DR Congo": "DR Congo", "Iran": "IR Iran",
}


def _resolve(name, ratings):
    if name in ratings:
        return name
    if name in ALIAS and ALIAS[name] in ratings:
        return ALIAS[name]
    # case-insensitive fallback
    low = {k.lower(): k for k in ratings}
    return low.get(name.lower())


def full_lambdas(pred, home, away, neutral):
    key = (home, away, bool(neutral))
    if key not in _LAMBDA:
        r = pred.predict_match(home, away, neutral=neutral)
        _LAMBDA[key] = (r["lambdas"][0], r["lambdas"][1])
    return _LAMBDA[key]


# ---------------------------- rendering ----------------------------

def _board_grid_html(board):
    out = ["<div class='boards'>"]
    for grpd in board:
        out.append("<div class='mkt'><h4>" + _html.escape(grpd["title"]) + "</h4><table>")
        out.append("".join(_row_html(row) for row in grpd["rows"]))
        out.append("</table></div>")
    out.append("</div>")
    return "".join(out)


def _inplay_card(r, board, league=""):
    h, a = r["home"], r["away"]
    sh, sa = r["score"]
    minute = int(r["minute"])
    pick = {"home": h, "draw": "Draw", "away": a}[r["outcome"]]
    lg = f" &middot; {_html.escape(league)}" if league else ""
    parts = ["<div class='game live'>"]
    parts.append("<div class='ghead'>")
    parts.append(f"<span class='pill livep'>● LIVE {minute}'</span>")
    parts.append(f"<span class='teams'>{_html.escape(h)} <span class='score'>{sh}–{sa}</span> {_html.escape(a)}</span>")
    parts.append(f"<span class='pill {r['outcome']}'>{_html.escape(pick)} {r['confidence_top']*100:.0f}%</span>")
    parts.append(f"<span class='meta'>proj final {r['proj_final'][0]}–{r['proj_final'][1]}"
                 f" &middot; exp {r['exp_final_home']:.2f}–{r['exp_final_away']:.2f}{lg}</span>")
    parts.append("</div>")
    parts.append(_edge_panel(r, board))
    parts.append(_board_grid_html(board))
    parts.append("</div>")
    return "".join(parts)


# ---- Kalshi edge overlay (live market prices, no auth) ----
_LIVE = {"ts": -1e9, "data": []}


def _cached_live(ttl=55.0):
    """api-football live fixtures, cached so page refreshes don't burn the 100/day quota."""
    now = time.monotonic()
    if now - _LIVE["ts"] >= ttl:
        try:
            _LIVE.update(ts=now, data=_lineups.live_matches())
        except Exception:
            _LIVE.update(ts=now, data=[])
    return _LIVE["data"]


def _entry_sides(board):
    for g in board:
        if g["title"].startswith("Match Result"):
            rws = g["rows"]
            return {"home": rws[0]["entry_c"], "draw": rws[1]["entry_c"], "away": rws[2]["entry_c"]}
    return {"home": None, "draw": None, "away": None}


_FLAG = {"GO": "★ GO", "value": "✓ value", "no edge": "·", "no market": "—"}


_STATMAP = {"GO": ("go", "★ GO"), "tentative": ("val", "? tentative"),
            "blocked": ("blk", "⛔ blocked"), "value": ("val", "✓ value"),
            "no edge": ("", "·"), "no market": ("", "—")}


def _edge_panel(r, board):
    """Model fair vs SHARP anchor vs live Kalshi -> fee-aware net edge with the discipline veto."""
    try:
        prices = _kalshi.match_winner_prices(r["home"], r["away"])
    except Exception:
        prices = None
    try:
        anchor = _sharp.anchor_fair(r["home"], r["away"])
    except Exception:
        anchor = None
    entry = _entry_sides(board)
    rows = _edges.disciplined_table(r, prices, entry, anchor)
    has_price = any(row["price_c"] is not None for row in rows)
    ev_ticker = (prices or {}).get("event", "") if prices else ""
    asrc = (f"sharp: {anchor['source']}" if anchor else "sharp: none (model-only)")
    head = ("<div class='edge'><div class='ehead'>⟂ Model vs <b>Sharp</b> vs <b>Kalshi</b>"
            + (f" &middot; <span class='tk'>{_html.escape(ev_ticker)}</span>" if ev_ticker else "")
            + f" &middot; {asrc} &middot; net of fees</div>")
    if not has_price:
        return head + "<div class='enote'>No live Kalshi price for this match yet.</div></div>"
    out = [head, "<table class='etbl'><tr><th>Outcome</th><th>Model</th><th>Sharp</th>"
           "<th>Kalshi</th><th>Net edge</th><th>EV</th><th></th></tr>"]
    for row in rows:
        cls, flag = _STATMAP.get(row["status"], ("", "·"))
        model = f"{row['model_c']:.0f}¢"
        sharp = f"{row['anchor_c']:.0f}¢" if row["anchor_c"] is not None else "—"
        price = f"{row['price_c']}¢" if row["price_c"] is not None else "—"
        ne = f"{row['net_edge_c']:+.1f}" if row["net_edge_c"] is not None else "—"
        ev = f"{row['ev']*100:+.0f}%" if row["ev"] is not None else "—"
        if row["price_c"] is not None:
            qs = urllib.parse.urlencode({"home": r["home"], "away": r["away"], "side": row["key"],
                                         "price": row["price_c"], "fair": round(row["cons_c"], 1)})
            flag += f" <a class='logbet' href='/paperbet?{qs}' title='paper-bet at {row['price_c']}¢'>📝</a>"
        out.append(f"<tr class='{cls}'><td>{_html.escape(row['label'])}</td>"
                   f"<td>{model}</td><td>{sharp}</td><td>{price}</td>"
                   f"<td>{ne}</td><td>{ev}</td><td class='eflag'>{flag}</td></tr>")
    out.append("</table></div>")
    return "".join(out)


def _game_card(r, board):
    """Pre-match card (cached model) + a freshly-priced Kalshi edge panel injected at the top."""
    card = _game_html(r, board)
    return card.replace("<div class='boards'>", _edge_panel(r, board) + "<div class='boards'>", 1)


def _ledger_html():
    """Paper-book scorecard + open positions (live P&L) + recent settled results."""
    try:
        _paper.settle()
    except Exception:
        pass
    rp = _paper.report()
    b = _paper._load()
    opens = [x for x in b["bets"] if x["status"] == "open"]
    settled = [x for x in b["bets"] if x["status"] == "settled"]
    pnl = rp["pnl"]; roi = rp["roi"]
    pc = "pos" if pnl >= 0 else "neg"
    summ = (f"{rp['settled']} settled &middot; realized <span class='{pc}'>"
            f"{'+' if pnl >= 0 else ''}${pnl:.2f}</span>")
    if roi is not None:
        summ += f" &middot; ROI <span class='{pc}'>{roi*100:+.0f}%</span>"
    if rp["win_rate"] is not None:
        summ += f" &middot; win {rp['win_rate']*100:.0f}%"
    if rp["avg_clv"] is not None:
        summ += f" &middot; CLV {rp['avg_clv']:+.1f}¢"
    summ += f" &middot; {len(opens)} open (${rp['open_stake']:.2f})"
    head = (f"<div class='ledger'><div class='lhead'>📒 Paper book <span class='lsum'>{summ}</span>"
            f"<a class='clearlink' href='/paperclear' title='reset the paper book'>clear</a></div>")
    if not opens and not settled:
        return head + "<div class='enote'>No paper bets yet — click 📝 on any edge row to log one (fake money).</div></div>"
    out = [head, "<table class='etbl ltbl'><tr><th>Bet</th><th>Entry</th>"
           "<th>Now / Result</th><th>P&amp;L</th></tr>"]
    for x in opens:
        lp = _paper.live_pnl(x)
        now = f"bid {lp['bid']}¢" if lp else "—"
        un = f"{'+' if lp['unrealized'] >= 0 else ''}${lp['unrealized']:.2f}" if lp else "—"
        uc = "pos" if (lp and lp["unrealized"] >= 0) else "neg"
        out.append(f"<tr><td>{x['contracts']}× {_html.escape(x['label'])}</td><td>{x['entry_c']}¢</td>"
                   f"<td>{now}</td><td class='{uc}'>{un}</td></tr>")
    for x in settled[-6:]:
        rc = "pos" if x["result"] == "WON" else "neg"
        pl = f"{'+' if x['pnl'] >= 0 else ''}${x['pnl']:.2f}"
        clv = f" &middot; CLV {x['clv_c']:+d}¢" if x.get("clv_c") is not None else ""
        out.append(f"<tr><td>{x['contracts']}× {_html.escape(x['label'])}</td><td>{x['entry_c']}¢</td>"
                   f"<td class='{rc}'>{x['result']}{clv}</td><td class='{rc}'>{pl}</td></tr>")
    out.append("</table></div>")
    return "".join(out)


@app.route("/paperbet")
def paperbet():
    from flask import redirect
    a = request.args
    try:
        _paper.place(a.get("home"), a.get("away"), a.get("side"), float(a.get("price")),
                     fair_c=(float(a.get("fair")) if a.get("fair") else None), note="dashboard")
    except Exception:
        pass
    return redirect("/")


@app.route("/paperclear")
def paperclear():
    from flask import redirect
    _paper._save({"bets": [], "next_id": 1})
    return redirect("/")


@app.route("/anchor")
def anchor():
    from flask import redirect
    a = request.args
    try:
        _sharp.set_manual(a.get("home"), a.get("away"), float(a.get("oh")),
                          float(a.get("od")), float(a.get("oa")))
    except Exception:
        pass
    return redirect("/")


@app.route("/anchorclear")
def anchorclear():
    from flask import redirect
    _sharp.clear_manual()
    return redirect("/")


def _manual_form(teams, vals):
    opts = "".join(f"<option value='{_html.escape(t)}'>" for t in teams)
    g = lambda k: _html.escape(str(vals.get(k, "")))
    return f"""
<form method='get' class='mform'>
  <span class='mlab'>What-if in-play:</span>
  <input list='teamlist' name='m_home' placeholder='Home team' value="{g('m_home')}" size='16'>
  <input class='sc' name='m_h' placeholder='0' value="{g('m_h')}" size='2'>
  <span>–</span>
  <input class='sc' name='m_a' placeholder='0' value="{g('m_a')}" size='2'>
  <input list='teamlist' name='m_away' placeholder='Away team' value="{g('m_away')}" size='16'>
  <span class='mlab'>minute</span>
  <input class='sc' name='m_min' placeholder='88' value="{g('m_min')}" size='3'>
  <label class='neu'><input type='checkbox' name='m_neutral' value='1' {'checked' if vals.get('m_neutral') else ''}> neutral</label>
  <button type='submit'>Price it</button>
  <datalist id='teamlist'>{opts}</datalist>
</form>"""


def _anchor_form(teams):
    opts = "".join(f"<option value='{_html.escape(t)}'>" for t in teams)
    return f"""
<form method='get' action='/anchor' class='mform aform'>
  <span class='mlab'>Sharp anchor — type Pinnacle decimal odds:</span>
  <input list='teamlist' name='home' placeholder='Home' size='13'>
  <span class='mlab'>@</span><input name='oh' class='sc' placeholder='2.10' size='4'>
  <span class='mlab'>draw</span><input name='od' class='sc' placeholder='3.30' size='4'>
  <input list='teamlist' name='away' placeholder='Away' size='13'>
  <span class='mlab'>@</span><input name='oa' class='sc' placeholder='3.90' size='4'>
  <button type='submit'>Set anchor</button>
  <a class='clearlink' href='/anchorclear' title='clear all anchors'>clear</a>
  <datalist id='teamlist'>{opts}</datalist>
</form>"""


_EXTRA_CSS = """
.live{border-color:#7a2230;} .pill.livep{background:#3a0e16;color:#ff6b81;}
.score{font-weight:800;font-size:20px;padding:0 8px;color:#fff;}
.bigrefresh{color:#8b97a6;font-size:12.5px;}
.mform{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#161d26;border:1px solid #2c3744;
border-radius:10px;padding:10px 14px;margin:0 0 18px;}
.mform input[type=text],.mform input:not([type]){background:#0f1419;color:#e6edf3;border:1px solid #2c3744;
border-radius:6px;padding:5px 8px;} .mform .sc{text-align:center;}
.mform button{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer;font-weight:600;}
.mform .mlab{color:#8b97a6;} .mform .neu{color:#8b97a6;font-size:12.5px;}
.section{margin:6px 0 4px;color:#cdd9e5;font-size:15px;font-weight:600;border-left:3px solid #2563eb;padding-left:9px;}
.empty{color:#8b97a6;font-size:13px;padding:6px 2px 14px;}
.edge{margin:0 18px;padding:10px 12px 6px;border-top:1px dashed #2c3744;}
.ehead{color:#cdd9e5;font-size:12.5px;margin-bottom:6px;} .ehead .tk{color:#6b7886;font-size:11px;}
.etbl{width:auto;border-collapse:collapse;font-size:12.5px;}
.etbl th{color:#8b97a6;text-align:right;font-weight:500;padding:2px 12px;border-bottom:1px solid #2c3744;}
.etbl th:first-child{text-align:left;}
.etbl td{padding:3px 12px;text-align:right;font-variant-numeric:tabular-nums;color:#c9d4df;}
.etbl td:first-child{text-align:left;color:#e6edf3;}
.etbl tr.go{background:#10301c;} .etbl tr.go .eflag{color:#74e08c;font-weight:700;}
.etbl tr.val{background:#2a2614;} .etbl tr.val .eflag{color:#e3c869;}
.etbl tr.blk{opacity:.5;} .etbl tr.blk .eflag{color:#c98b8b;}
.etbl .eflag{font-weight:600;}
.aform{margin-top:-10px;}
.enote{color:#8b97a6;font-size:12px;padding:2px 0 8px;}
.ledger{background:#13201a;border:1px solid #28402f;border-radius:10px;padding:10px 14px;margin:0 0 18px;}
.lhead{color:#cdd9e5;font-size:13px;font-weight:600;margin-bottom:4px;}
.lsum{color:#8b97a6;font-size:12px;font-weight:400;}
.ltbl{margin-top:6px;} .ltbl th{font-weight:500;}
.pos{color:#74e08c;} .neg{color:#f0838a;}
.clearlink{color:#6b7886;font-size:11px;margin-left:10px;text-decoration:none;}
.clearlink:hover{color:#c93c37;}
.logbet{text-decoration:none;font-size:12px;margin-left:5px;opacity:.7;} .logbet:hover{opacity:1;}
"""


def page_html(asof, roi, manual_card, live_cards, prematch_cards, teams, vals, key_on, ledger=""):
    head = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>WC2026 Live Fair-Value</title><style>{_CSS}{_EXTRA_CSS}</style></head><body>")
    hdr = ("<header><h1>⚽ WC2026 — Live Fair-Value Server</h1>"
           f"<div class='sub'>As of {asof.date()} &middot; refreshes every "
           f"<b>60 s</b> (<span id='cd'>60</span>s) &middot; “Fair” = model true prob in ¢ &middot; "
           f"“Enter ≤” clears {roi*100:.0f}% expected ROI &middot; "
           f"live feed: {'ON' if key_on else 'add API_FOOTBALL_KEY to auto-pull live games'}"
           f" &middot; Kalshi edges: ON</div></header>")
    body = ["<div class='wrap'>"]
    if ledger:
        body.append(ledger)
    body.append(_manual_form(teams, vals))
    body.append(_anchor_form(teams))
    if manual_card:
        body.append("<div class='section'>What-if in-play</div>")
        body.append(manual_card)
    body.append("<div class='section'>● Live now</div>")
    if live_cards:
        body.extend(live_cards)
    else:
        body.append("<div class='empty'>" + ("No games in play right now." if key_on else
                    "No live feed — add your api-football key (export API_FOOTBALL_KEY=…) to auto-pull "
                    "in-play games, or use the what-if tool above.") + "</div>")
    body.append("<div class='section'>Upcoming — pre-match fair value</div>")
    body.extend(prematch_cards)
    body.append("</div>")
    js = ("<script>let s=60;const el=document.getElementById('cd');"
          "setInterval(()=>{s--;if(el)el.textContent=s;"
          "if(s<=0){sessionStorage.setItem('sy',window.scrollY);location.reload();}},1000);"
          "window.addEventListener('load',()=>{const y=sessionStorage.getItem('sy');"
          "if(y)window.scrollTo(0,parseInt(y));});</script>")
    return head + hdr + "".join(body) + js + "</body></html>"


# ---------------------------- routes ----------------------------

@app.route("/")
def index():
    pred = STATE["pred"]; roi = STATE["roi"]
    ratings = pred.ratings

    # manual what-if in-play
    vals, manual_card = {}, None
    for k in ("m_home", "m_away", "m_h", "m_a", "m_min", "m_neutral"):
        v = request.args.get(k)
        if v not in (None, ""):
            vals[k] = v
    if vals.get("m_home") and vals.get("m_away"):
        rh = _resolve(vals["m_home"], ratings); ra = _resolve(vals["m_away"], ratings)
        if rh and ra:
            neutral = bool(vals.get("m_neutral"))
            lh, la = full_lambdas(pred, rh, ra, neutral)
            try:
                sh = int(vals.get("m_h") or 0); sa = int(vals.get("m_a") or 0)
                mn = float(vals.get("m_min") or 0)
            except ValueError:
                sh = sa = 0; mn = 0
            r, board = inplay_board(rh, ra, lh, la, mn, sh, sa, neutral=neutral, min_roi=roi)
            manual_card = _inplay_card(r, board)
        else:
            miss = vals["m_home"] if not rh else vals["m_away"]
            manual_card = f"<div class='empty'>Unknown team: “{_html.escape(miss)}”. Try the exact model name.</div>"

    # live games from api-football (cached to protect the 100/day quota)
    live_cards = []
    for lm in _cached_live():
        rh = _resolve(lm["home"], ratings); ra = _resolve(lm["away"], ratings)
        if not (rh and ra):
            continue
        lh, la = full_lambdas(pred, rh, ra, True)   # WC games are neutral venues
        r, board = inplay_board(rh, ra, lh, la, lm["minute"], lm["score_h"], lm["score_a"],
                                neutral=True, min_roi=roi)
        live_cards.append(_inplay_card(r, board, league=lm.get("league", "")))

    # pre-match cards rendered fresh each request so Kalshi prices update (model itself is cached)
    prematch_cards = [_game_card(r, board) for (r, board) in STATE["prematch_rb"]]
    return page_html(pred.asof, roi, manual_card, live_cards, prematch_cards,
                     STATE["teams"], vals, _lineups.available(), ledger=_ledger_html())


def build_state(days, roi, asof, rebuild):
    pred = Predictor(asof=asof, rebuild_table=rebuild)
    up = upcoming_matches(pred.df).sort_values("date")
    lo = pred.asof.normalize(); hi = lo + pd.Timedelta(days=days)
    fx = up[(up["date"] >= lo) & (up["date"] < hi)]
    prematch = []
    for _, m in fx.iterrows():
        r = pred.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]),
                               date=m["date"])
        prematch.append((r, market_board(r, min_roi=roi)))
    teams = sorted(set(up["home_team"]) | set(up["away_team"]))
    STATE.update(pred=pred, roi=roi, prematch_rb=prematch, teams=teams)
    print(f"[server] ready: {len(prematch)} upcoming fixtures cached, {len(teams)} teams known")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--roi", type=float, default=MIN_ROI)
    ap.add_argument("--date", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    print("[server] building model (one-time) ...")
    build_state(args.days, args.roi, args.date, args.rebuild)
    print(f"\n  ➜  open  http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
