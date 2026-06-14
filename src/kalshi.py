"""Kalshi live market prices for World Cup match markets — NO AUTH required (public read-only data).

Kalshi organises World Cup match-winners under series KXWCGAME, one event per fixture:
    KXWCGAME-26JUN13BRAMAR   "Brazil vs Morocco"
with three markets, each tagged by full team name in yes_sub_title:
    ...-BRA (yes=Brazil)   ...-MAR (yes=Morocco)   ...-TIE (yes=Tie/Draw)

A binary contract's price in cents already equals an implied probability, so to BUY the "Brazil wins"
outcome you pay the YES ask = 100 - (best NO bid). That ask, compared to our model's fair value, is the
edge. We never trade here — we only read prices so the dashboard can show where value is.

Other useful series (same pattern) for later: KXWCTOTAL (totals), KXWCBTTS, KXWCFTTS (first to score),
KXWCSCORE (correct score), KXWCSPREAD, KXWC1H* (first half), KXMENWORLDCUP (tournament winner).
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.elections.kalshi.com/trade-api/v2"

_EVENTS = {}                              # status -> (ts, [events])  KXWCGAME cache (5 min)
_OB = {}                                  # ticker -> (ts, price dict)  short cache
_EVENTS_TTL = 300.0
_OB_TTL = 20.0

# Kalshi team name -> our model name (extend as needed)
ALIAS = {"Tie": "Draw", "USA": "United States", "South Korea": "Korea Republic",
         "Ivory Coast": "Côte d'Ivoire", "Turkiye": "Türkiye", "Turkey": "Türkiye",
         "Czechia": "Czechia", "Iran": "IR Iran", "Cape Verde": "Cabo Verde"}


def _get(path, params=None, timeout=6):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def available():
    """True if Kalshi market data is reachable."""
    return _get("/markets", {"limit": 1}) is not None


def _norm(name):
    n = (name or "").strip()
    return ALIAS.get(n, n)


def _events(status):
    now = time.monotonic()
    c = _EVENTS.get(status)
    if c and now - c[0] < _EVENTS_TTL:
        return c[1]
    out, cursor = [], None
    for _ in range(4):  # a few pages is plenty for one tournament
        params = {"series_ticker": "KXWCGAME", "status": status,
                  "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/events", params)
        if not d:
            break
        out.extend(d.get("events", []))
        cursor = d.get("cursor")
        if not cursor:
            break
    _EVENTS[status] = (now, out)
    return out


def _game_events():
    # open first (the common case) then settled/closed so finished games still resolve
    return _events("open") + _events("settled") + _events("closed")


def find_match(home, away):
    """Return {'home': ticker, 'draw': ticker, 'away': ticker} for a fixture, or None if not listed."""
    want = {home, away}
    for e in _game_events():
        by_name = {}
        for m in e.get("markets", []):
            nm = _norm(m.get("yes_sub_title"))
            if nm:
                by_name[nm] = m.get("ticker")
        teams = set(by_name) - {"Draw"}
        if want <= teams:
            res = {}
            if home in by_name:
                res["home"] = by_name[home]
            if away in by_name:
                res["away"] = by_name[away]
            if "Draw" in by_name:
                res["draw"] = by_name["Draw"]
            res["event"] = e.get("event_ticker")
            return res
    return None


def _best(arr):
    """Best (highest) bid price in cents from a Kalshi orderbook side ([[price,size],...])."""
    bp, _ = _best2(arr)
    return bp


def _best2(arr):
    """(best_price_cents, size_at_that_level) from a Kalshi orderbook side."""
    best, size = None, None
    for row in arr or []:
        try:
            p = int(round(float(row[0]) if float(row[0]) > 1.5 else float(row[0]) * 100))
            s = float(row[1]) if len(row) > 1 else None
        except Exception:
            continue
        if best is None or p > best:
            best, size = p, s
    return best, size


def taker_fee(price_c, contracts, mult=0.07):
    """Kalshi taker fee in DOLLARS = ceil(mult * C * p * (1-p)) to the next cent. p = price in dollars."""
    import math
    p = price_c / 100.0
    return math.ceil(mult * contracts * p * (1 - p) * 100) / 100.0


def yes_price(ticker):
    """Cost in cents to BUY one YES contract now (=100-best_no_bid), plus the yes bid and a mid.

    Returns dict {ask, bid, mid, last, volume} (any may be None if no resting liquidity)."""
    now = time.monotonic()
    cached = _OB.get(ticker)
    if cached and now - cached[0] < _OB_TTL:
        return cached[1]
    d = _get(f"/markets/{ticker}/orderbook")
    out = dict(ask=None, bid=None, mid=None, last=None, volume=None, ask_size=None, bid_size=None)
    if d:
        ob = d.get("orderbook") or {}
        obfp = d.get("orderbook_fp") or {}          # Kalshi nests resting orders here (dollar strings)
        yes_side = ob.get("yes") or obfp.get("yes_dollars")
        no_side = ob.get("no") or obfp.get("no_dollars")
        yes_bid, bid_size = _best2(yes_side)
        no_bid, no_size = _best2(no_side)
        ask = (100 - no_bid) if no_bid is not None else None
        mid = None
        if yes_bid is not None and ask is not None:
            mid = (yes_bid + ask) / 2.0
        elif ask is not None:
            mid = ask
        elif yes_bid is not None:
            mid = yes_bid
        # ask_size = how many YES you can buy at the ask (= size resting on the best NO bid)
        out.update(ask=ask, bid=yes_bid, mid=mid, ask_size=no_size, bid_size=bid_size)
    _OB[ticker] = (now, out)
    return out


def market_result(ticker):
    """Settlement info for a market: {settled, result('yes'/'no'/''), last_c, status}. None on error."""
    d = _get(f"/markets/{ticker}")
    if not d:
        return None
    m = d.get("market") or {}
    result = m.get("result") or ""
    status = m.get("status") or ""
    settled = result in ("yes", "no") or status in ("settled", "finalized", "determined")
    last = m.get("last_price")
    try:
        last = int(round(float(last))) if last is not None else None
    except Exception:
        last = None
    return dict(settled=settled, result=result, last_c=last, status=status)


def match_winner_prices(home, away):
    """{'home':{ask,bid,mid}, 'draw':..., 'away':...} of live Kalshi prices, or None if unlisted."""
    mk = find_match(home, away)
    if not mk:
        return None
    out = {"event": mk.get("event")}
    for side in ("home", "draw", "away"):
        if side in mk:
            out[side] = yes_price(mk[side])
    return out


if __name__ == "__main__":
    print("Kalshi reachable:", available())
    for h, a in [("Brazil", "Morocco"), ("Scotland", "Brazil"), ("United States", "Paraguay")]:
        p = match_winner_prices(h, a)
        if not p:
            print(f"  {h} vs {a}: not listed on Kalshi")
            continue
        print(f"  {h} vs {a}  (event {p.get('event')})")
        for side in ("home", "draw", "away"):
            if side in p:
                q = p[side]
                print(f"     {side:5s}: ask {q['ask']}  bid {q['bid']}  mid {q['mid']}")
