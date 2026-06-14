"""Mock $1000 portfolio with fractional-KELLY sizing — simulates real risk management.

Flat-stake paper trading tells you if a signal wins; it does NOT tell you how to SIZE. Kelly does.
For a binary contract bought at price c (dollars) with fair win-prob p, the growth-optimal fraction of
bankroll is the classic

    f* = (p - c) / (1 - c)

We use FRACTIONAL Kelly (¼) — because our edge estimate is itself uncertain, quarter-Kelly is the
standard hedge — and cap any single bet at 10% of bankroll. Stakes scale with the live bankroll, so
the sim compounds: a cold streak shrinks bet sizes, a hot streak grows them. Fees are charged at entry
and P&L marks to the live Kalshi bid; positions settle off the real Kalshi result.

State: outputs/portfolio.json  (persistent).  CLI:
    python src/portfolio.py update      # settle finished + open new Kelly-sized positions
    python src/portfolio.py report      # equity, return, drawdown
    python src/portfolio.py reset
"""
import argparse
import json
from datetime import datetime

from config import OUTPUTS
import kalshi

PORT_PATH = OUTPUTS / "portfolio.json"
START = 1000.0
KELLY = 0.25          # fractional Kelly
MAX_FRACTION = 0.10   # hard cap: never risk >10% of bankroll on one contract line
LABELS = {"home": "{home} win", "draw": "Draw", "away": "{away} win"}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _load():
    if PORT_PATH.exists():
        try:
            return json.loads(PORT_PATH.read_text())
        except Exception:
            pass
    return {"start": START, "cash": START, "positions": [], "equity_curve": [], "next_id": 1}


def _save(p):
    PORT_PATH.write_text(json.dumps(p, indent=2))


def kelly_fraction(p, c):
    """Full-Kelly bankroll fraction for a binary contract: (p - c)/(1 - c), floored at 0."""
    if c <= 0 or c >= 1:
        return 0.0
    return max(0.0, (p - c) / (1.0 - c))


def size_contracts(fair_prob, price_c, bankroll, kelly=KELLY, cap=MAX_FRACTION):
    c = price_c / 100.0
    f = min(cap, kelly * kelly_fraction(fair_prob, c))
    n = int((f * bankroll) // c) if c > 0 else 0
    return n, f


def equity(p):
    """Cash + mark-to-market of open positions at the current Kalshi bid."""
    mtm = 0.0
    for pos in p["positions"]:
        if pos["status"] == "open":
            bid = None
            if pos.get("ticker"):
                bid = kalshi.yes_price(pos["ticker"]).get("bid")
            mtm += pos["contracts"] * ((bid if bid is not None else pos["entry_c"]) / 100.0)
    return p["cash"] + mtm


def settle(p):
    n = 0
    for pos in p["positions"]:
        if pos["status"] != "open" or not pos.get("ticker"):
            continue
        info = kalshi.market_result(pos["ticker"])
        if not info or not info.get("settled"):
            continue
        won = info["result"] == "yes"
        payout = pos["contracts"] * 1.0 if won else 0.0
        p["cash"] += payout
        pos["pnl"] = round(payout - pos["stake"] - pos["fee"], 2)
        pos["result"] = "WON" if won else "LOST"
        pos["status"] = "settled"
        n += 1
    return n


def open_position(home, away, side, price_c, fair_prob, ticker, p):
    bank = equity(p)
    n, f = size_contracts(fair_prob, price_c, bank)
    if n < 1:
        return None
    c = price_c / 100.0
    stake = round(n * c, 2)
    fee = round(kalshi.taker_fee(price_c, n), 2)
    if stake + fee > p["cash"]:                      # respect available cash
        n = int((p["cash"] * 0.98) // c)
        if n < 1:
            return None
        stake = round(n * c, 2); fee = round(kalshi.taker_fee(price_c, n), 2)
    p["cash"] -= (stake + fee)
    pos = dict(id=p["next_id"], ts=_now(), home=home, away=away, side=side,
               label=LABELS[side].format(home=home, away=away), ticker=ticker, entry_c=price_c,
               contracts=n, stake=stake, fee=fee, kelly_f=round(f, 4),
               fair_c=round(fair_prob * 100, 1), status="open", result=None, pnl=None)
    p["positions"].append(pos)
    p["next_id"] += 1
    return pos


def update(days=4, pred=None, tentative=True):
    """Settle finished positions, then open Kelly-sized positions for fresh signals."""
    import paper
    p = _load()
    settle(p)
    existing = {(x["ticker"], x["entry_c"]) for x in p["positions"]}
    opened = []
    for s in paper.current_signals(days, pred):
        ok = s["go"] or (tentative and s["status"] == "tentative")
        if not (ok and s["price_c"] and s["ticker"]):
            continue
        if (s["ticker"], s["price_c"]) in existing:
            continue
        pos = open_position(s["home"], s["away"], s["side"], s["price_c"],
                            s["fair_c"] / 100.0, s["ticker"], p)
        if pos:
            opened.append(pos)
            existing.add((s["ticker"], s["price_c"]))
    p["equity_curve"].append({"ts": _now(), "equity": round(equity(p), 2)})
    _save(p)
    return opened


def report(p=None):
    if p is None:
        p = _load()
    eq = equity(p)
    settled = [x for x in p["positions"] if x["status"] == "settled"]
    opens = [x for x in p["positions"] if x["status"] == "open"]
    realized = sum(x["pnl"] for x in settled if x["pnl"] is not None)
    wins = sum(1 for x in settled if x["result"] == "WON")
    peak, mdd = p["start"], 0.0
    for e in p["equity_curve"]:
        peak = max(peak, e["equity"])
        mdd = min(mdd, (e["equity"] - peak) / peak)
    return dict(start=p["start"], equity=round(eq, 2), cash=round(p["cash"], 2),
                ret=(eq - p["start"]) / p["start"], realized=round(realized, 2),
                n_settled=len(settled), n_open=len(opens),
                win_rate=(wins / len(settled) if settled else None),
                max_dd=mdd, exposure=round(eq - p["cash"], 2))


def _print(p=None):
    rp = report(p)
    print("\n========  KELLY $1000 PORTFOLIO  ========")
    print(f"  equity ${rp['equity']:.2f}  (start ${rp['start']:.0f}, "
          f"return {rp['ret']*100:+.1f}%)   cash ${rp['cash']:.2f}")
    print(f"  positions: {rp['n_settled']} settled, {rp['n_open']} open  "
          f"(${rp['exposure']:.2f} at risk)")
    if rp["win_rate"] is not None:
        print(f"  realized P&L ${rp['realized']:+.2f}   win rate {rp['win_rate']*100:.0f}%")
    print(f"  max drawdown {rp['max_dd']*100:.1f}%")
    print("=========================================")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    pu = sub.add_parser("update"); pu.add_argument("--days", type=int, default=4)
    sub.add_parser("report"); sub.add_parser("reset")
    args = ap.parse_args()
    if args.cmd == "update":
        opened = update(days=args.days)
        print(f"opened {len(opened)} Kelly-sized position(s):")
        for x in opened:
            print(f"  {x['contracts']}× {x['label']} @ {x['entry_c']}¢  "
                  f"(¼-Kelly {x['kelly_f']*100:.1f}% -> ${x['stake']:.2f})")
        _print()
    elif args.cmd == "reset":
        _save({"start": START, "cash": START, "positions": [], "equity_curve": [], "next_id": 1})
        print(f"portfolio reset to ${START:.0f}.")
    else:
        p = _load(); settle(p); _save(p); _print(p)


if __name__ == "__main__":
    main()
