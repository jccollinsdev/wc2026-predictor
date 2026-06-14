"""Paper-trading book for Kalshi World Cup match-winner bets — fake money, real prices, real settlement.

You log a bet at the live Kalshi price; it tracks live P&L against the current bid; and when the match
settles on Kalshi the bet auto-settles (WON -> contracts*$1, LOST -> $0). The scorecard tracks the only
things that tell you if the model's edge is real over many bets:
    realized ROI, win rate, and CLV (closing-line value: did your entry beat the final market price?).

Storage: outputs/paper_book.json. Source-of-truth for settlement = the Kalshi market's own result
(no api-football quota used).

CLI:
    python src/paper.py bet Brazil Morocco away 21 100   # buy 100 "Morocco win" @ 21c (fake)
    python src/paper.py bet Brazil Morocco away 21        # stake defaults to $10 -> contracts auto
    python src/paper.py book                              # open positions + live P&L
    python src/paper.py settle                            # auto-settle any finished matches
    python src/paper.py report                            # the scorecard
    python src/paper.py auto                              # paper-bet EVERY current model GO signal
"""
import argparse
import json
from datetime import datetime

from config import OUTPUTS
import kalshi

BOOK_PATH = OUTPUTS / "paper_book.json"
DEFAULT_STAKE = 10.0          # $ per bet when contracts not given (flat-stake = cleanest edge test)
SIDE_LABELS = {"home": "{home} win", "draw": "Draw", "away": "{away} win"}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _load():
    if BOOK_PATH.exists():
        try:
            return json.loads(BOOK_PATH.read_text())
        except Exception:
            pass
    return {"bets": [], "next_id": 1}


def _save(b):
    BOOK_PATH.write_text(json.dumps(b, indent=2))


def place(home, away, side, price_c, contracts=None, fair_c=None, note="", stake=DEFAULT_STAKE):
    """Log a paper bet. side in {home,draw,away}. Returns the bet dict (or {'error':..})."""
    side = side.lower()
    if side not in SIDE_LABELS:
        return {"error": f"side must be home/draw/away, got {side!r}"}
    price_c = int(round(float(price_c)))
    if not (1 <= price_c <= 99):
        return {"error": f"price must be 1-99 cents, got {price_c}"}
    if contracts is None:
        contracts = max(1, round(stake * 100 / price_c))
    contracts = int(contracts)
    mk = kalshi.find_match(home, away)
    ticker = (mk or {}).get(side)
    fee = round(kalshi.taker_fee(price_c, contracts), 2)   # Kalshi taker fee modeled at entry
    b = _load()
    bet = dict(id=b["next_id"], ts=_now(), home=home, away=away, side=side,
               label=SIDE_LABELS[side].format(home=home, away=away), ticker=ticker,
               entry_c=price_c, contracts=contracts, stake=round(contracts * price_c / 100.0, 2),
               fee=fee, fair_c=(round(float(fair_c), 1) if fair_c is not None else None),
               note=note, status="open", result=None, pnl=None, close_c=None, clv_c=None)
    b["bets"].append(bet)
    b["next_id"] += 1
    _save(b)
    return bet


def settle():
    """Auto-settle any open bets whose Kalshi market has resolved. Returns count settled."""
    b = _load()
    n = 0
    for bet in b["bets"]:
        if bet["status"] != "open" or not bet.get("ticker"):
            continue
        info = kalshi.market_result(bet["ticker"])
        if not info or not info.get("settled"):
            continue
        won = info["result"] == "yes"
        e, c = bet["entry_c"], bet["contracts"]
        bet["pnl"] = round(c * ((100 - e) if won else -e) / 100.0 - bet.get("fee", 0.0), 2)  # net of fee
        bet["result"] = "WON" if won else "LOST"
        bet["close_c"] = info.get("last_c")
        if info.get("last_c") is not None:
            bet["clv_c"] = info["last_c"] - e   # >0 = you bought below the closing price (good)
        bet["status"] = "settled"
        n += 1
    if n:
        _save(b)
    return n


def live_pnl(bet):
    """Unrealized P&L for an open bet at the current Kalshi bid (what you could sell for now)."""
    if not bet.get("ticker"):
        return None
    q = kalshi.yes_price(bet["ticker"])
    bid = q.get("bid")
    if bid is None:
        return None
    val = bet["contracts"] * bid / 100.0
    return dict(bid=bid, value=round(val, 2), unrealized=round(val - bet["stake"], 2))


def report():
    b = _load()
    settled = [x for x in b["bets"] if x["status"] == "settled"]
    open_ = [x for x in b["bets"] if x["status"] == "open"]
    staked = sum(x["stake"] for x in settled)
    pnl = sum(x["pnl"] for x in settled if x["pnl"] is not None)
    wins = sum(1 for x in settled if x["result"] == "WON")
    clvs = [x["clv_c"] for x in settled if x["clv_c"] is not None]
    fees = sum(x.get("fee", 0.0) for x in b["bets"])
    return dict(
        total=len(b["bets"]), settled=len(settled), open=len(open_),
        staked=round(staked, 2), pnl=round(pnl, 2), fees=round(fees, 2),
        roi=(pnl / staked if staked else None),
        win_rate=(wins / len(settled) if settled else None),
        avg_clv=(sum(clvs) / len(clvs) if clvs else None),
        open_stake=round(sum(x["stake"] for x in open_), 2),
    )


# ---------- auto: paper-bet every current model GO signal ----------

def current_signals(days=3, pred=None):
    """Disciplined match-winner signals across upcoming games (shared by paper + portfolio + trackers).

    Returns a list of dicts: home, away, date, side, label, price_c, fair_c (consensus), model_c,
    status (GO/tentative/blocked/value/no edge/no market), go (bool), ticker, event. Builds one
    Predictor unless one is passed in."""
    import pandas as pd
    from markets import market_board
    from data import upcoming_matches
    import edges
    import sharp
    if pred is None:
        from predict import Predictor
        pred = Predictor(verbose=False)
    up = upcoming_matches(pred.df).sort_values("date")
    lo = pred.asof.normalize(); hi = lo + pd.Timedelta(days=days)
    fx = up[(up["date"] >= lo) & (up["date"] < hi)]
    sig = []
    for _, m in fx.iterrows():
        r = pred.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]), date=m["date"])
        board = market_board(r)
        entry = {"home": None, "draw": None, "away": None}
        for g in board:
            if g["title"].startswith("Match Result"):
                entry = {"home": g["rows"][0]["entry_c"], "draw": g["rows"][1]["entry_c"],
                         "away": g["rows"][2]["entry_c"]}
        prices = kalshi.match_winner_prices(r["home"], r["away"])
        if not prices:
            continue
        anchor = sharp.anchor_fair(r["home"], r["away"])
        mk = kalshi.find_match(r["home"], r["away"])
        for row in edges.disciplined_table(r, prices, entry, anchor):
            sig.append(dict(home=r["home"], away=r["away"], date=str(m["date"].date()),
                            side=row["key"], label=row["label"], price_c=row["price_c"],
                            fair_c=row["cons_c"], model_c=row["model_c"], status=row["status"],
                            go=row["go"], ticker=(mk or {}).get(row["key"]),
                            event=(mk or {}).get("event")))
    return sig


def auto(days=3, stake=DEFAULT_STAKE, tentative=False, pred=None):
    """Paper-bet the model's signals, DISCIPLINED by the sharp anchor. Only confirmed GO by default;
    tentative=True also logs model-only signals where no anchor exists yet."""
    existing = {(x["ticker"], x["entry_c"]) for x in _load()["bets"]}
    logged, blocked = [], 0
    for s in current_signals(days, pred):
        if s["status"] == "blocked":
            blocked += 1
        ok = s["go"] or (tentative and s["status"] == "tentative")
        if ok and s["price_c"]:
            if (s["ticker"], s["price_c"]) in existing:
                continue
            bet = place(s["home"], s["away"], s["side"], s["price_c"], fair_c=s["fair_c"],
                        note=("auto-GO" if s["go"] else "auto-tentative"), stake=stake)
            if "error" not in bet:
                logged.append(bet)
                existing.add((s["ticker"], s["price_c"]))
    return logged, blocked


def _print_report():
    settle()
    rp = report()
    print("\n================  PAPER SCORECARD  ================")
    print(f"  bets: {rp['total']}  (settled {rp['settled']}, open {rp['open']})")
    print(f"  settled stake ${rp['staked']:.2f}  ->  realized P&L "
          f"{'+' if rp['pnl'] >= 0 else ''}${rp['pnl']:.2f}")
    if rp["roi"] is not None:
        print(f"  ROI: {rp['roi']*100:+.1f}%   win rate: {rp['win_rate']*100:.0f}%")
    if rp["avg_clv"] is not None:
        print(f"  avg CLV: {rp['avg_clv']:+.1f}¢  (>0 means you beat the closing price -> real edge)")
    print(f"  fees modeled: ${rp['fees']:.2f}   open exposure: ${rp['open_stake']:.2f}")
    print("==================================================")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pb = sub.add_parser("bet"); pb.add_argument("home"); pb.add_argument("away")
    pb.add_argument("side"); pb.add_argument("price", type=float)
    pb.add_argument("contracts", nargs="?", type=int, default=None)
    sub.add_parser("book"); sub.add_parser("settle"); sub.add_parser("report")
    sub.add_parser("clear")
    pa = sub.add_parser("auto"); pa.add_argument("--days", type=int, default=3)
    pa.add_argument("--tentative", action="store_true", help="also log model-only signals (no anchor)")
    args = ap.parse_args()

    if args.cmd == "bet":
        bet = place(args.home, args.away, args.side, args.price, args.contracts)
        if "error" in bet:
            print("ERROR:", bet["error"]); return
        tk = bet["ticker"] or "(no Kalshi market found — will not auto-settle)"
        print(f"logged #{bet['id']}: {bet['contracts']} x {bet['label']} @ {bet['entry_c']}¢  "
              f"stake ${bet['stake']:.2f}  | {tk}")
    elif args.cmd == "book":
        settle()
        b = _load()
        opens = [x for x in b["bets"] if x["status"] == "open"]
        print(f"\nOPEN PAPER POSITIONS ({len(opens)}):")
        for x in opens:
            lp = live_pnl(x)
            now = (f"bid {lp['bid']}¢  unreal {'+' if lp['unrealized'] >= 0 else ''}${lp['unrealized']:.2f}"
                   if lp else "no live quote")
            fair = f" fair {x['fair_c']}¢" if x["fair_c"] else ""
            print(f"  #{x['id']:>2} {x['contracts']:>4} x {x['label']:<22} @ {x['entry_c']}¢{fair}  ->  {now}")
        _print_report()
    elif args.cmd == "settle":
        print(f"settled {settle()} bet(s).")
        _print_report()
    elif args.cmd == "clear":
        _save({"bets": [], "next_id": 1})
        print("paper book cleared.")
    elif args.cmd == "auto":
        logged, blocked = auto(days=args.days, tentative=args.tentative)
        print(f"auto-logged {len(logged)} signal(s); sharp anchor BLOCKED {blocked} model signal(s).")
        for x in logged:
            print(f"  #{x['id']} {x['contracts']} x {x['label']} @ {x['entry_c']}¢ "
                  f"(fair {x['fair_c']}¢) [{x['note']}]")
        if not logged and not args.tentative:
            print("  (no confirmed GO — add a sharp anchor, or use --tentative for model-only signals)")
        _print_report()
    else:
        _print_report()


if __name__ == "__main__":
    main()
