"""Turn model fair value + a live market price into an EDGE.

For a binary (Kalshi-style) contract, the price in cents IS the implied probability, so:
    EV per contract (in cents) = fair_cents - price
    expected ROI on stake      = (fair_prob - price/100) / (price/100)
    GO (positive value)        = price <= entry_cents   (entry already bakes in the MIN_ROI buffer)

This is source-agnostic: the price can come from Kalshi (kalshi.py) or be typed in by hand. We only
flag value; you place the trades.
"""


def edge_for(fair_prob, entry_c, price_c):
    """fair_prob in [0,1]; entry_c, price_c in cents. Returns an edge dict (None price -> no_market)."""
    fair_c = fair_prob * 100.0
    if price_c is None:
        return dict(price_c=None, edge_c=None, ev_pct=None, go=False, status="no market")
    edge_c = fair_c - price_c
    ev_pct = ((fair_prob - price_c / 100.0) / (price_c / 100.0)) if price_c > 0 else None
    go = (entry_c is not None) and (price_c <= entry_c)
    status = "GO" if go else ("value" if edge_c > 0 else "no edge")
    return dict(price_c=price_c, edge_c=edge_c, ev_pct=ev_pct, go=go, status=status)


ANCHOR_BLEND = 0.5     # consensus weight: model vs sharp anchor
MIN_EDGE_PP = 1.0      # anchor must see at least this much value (pp) to confirm a GO


def disciplined_table(r, prices, entry_by_side, anchor):
    """Model fair vs sharp ANCHOR vs Kalshi, with fees, and a discipline veto.

    anchor: {'home':p,'draw':p,'away':p,'source':..} de-vigged sharp probs, or None.
    GO requires BOTH the model AND the sharp anchor to see net (after-fee) value vs the Kalshi ask.
    If the model sees value but the anchor agrees with Kalshi -> 'blocked' (this is the whole point).
    If no anchor is available -> 'tentative' (model-only, not a confirmed GO).
    """
    import kalshi
    sides = [("home", r["home"] + " win", r["p_home"]),
             ("draw", "Draw", r["p_draw"]),
             ("away", r["away"] + " win", r["p_away"])]
    rows = []
    for key, label, pm in sides:
        q = (prices or {}).get(key) if prices else None
        price_c = q.get("ask") if q else None
        anc = anchor.get(key) if anchor else None        # sharp fair prob (0..1) for this outcome
        cons = (ANCHOR_BLEND * pm + (1 - ANCHOR_BLEND) * anc) if anc is not None else pm
        entry = entry_by_side.get(key)
        fee_c = (kalshi.taker_fee(price_c, 1) * 100) if price_c else 0.0   # cents per contract
        rec = dict(key=key, label=label, model_c=pm * 100,
                   anchor_c=(anc * 100 if anc is not None else None), cons_c=cons * 100,
                   price_c=price_c, fee_c=fee_c, bid=(q or {}).get("bid"),
                   ask_size=(q or {}).get("ask_size"))
        if price_c is None:
            rec.update(net_edge_c=None, ev=None, status="no market", go=False)
            rows.append(rec); continue
        net_edge = cons * 100 - price_c - fee_c          # after-fee edge vs consensus fair
        ev = (net_edge / price_c) if price_c > 0 else None
        model_sees = (pm * 100 - price_c) > MIN_EDGE_PP
        cheap = entry is not None and price_c <= entry
        if anc is None:
            status = "tentative" if (cheap and net_edge > 0) else ("value" if net_edge > 0 else "no edge")
            go = False
        else:
            anchor_sees = (anc * 100 - price_c) > MIN_EDGE_PP
            if model_sees and not anchor_sees:
                status, go = "blocked", False
            elif cheap and net_edge > 0 and anchor_sees:
                status, go = "GO", True
            elif net_edge > 0:
                status, go = "value", False
            else:
                status, go = "no edge", False
        rec.update(net_edge_c=net_edge, ev=ev, status=status, go=go)
        rows.append(rec)
    return rows


def match_winner_table(r, prices, entry_by_side):
    """Build rows comparing model fair value vs live Kalshi for home/draw/away.

    r: predict_match()/inplay_board() result (has p_home/p_draw/p_away, home, away).
    prices: kalshi.match_winner_prices() output (or None).
    entry_by_side: {'home':entry_c,'draw':entry_c,'away':entry_c} from the model's market board.
    Returns list of dicts: {label, fair_prob, fair_c, price_c, edge_c, ev_pct, go, status}.
    """
    sides = [("home", r["home"] + " win", r["p_home"]),
             ("draw", "Draw", r["p_draw"]),
             ("away", r["away"] + " win", r["p_away"])]
    rows = []
    for key, label, prob in sides:
        q = (prices or {}).get(key) if prices else None
        price_c = q.get("ask") if q else None          # cost to BUY this outcome now
        e = edge_for(prob, entry_by_side.get(key), price_c)
        rows.append(dict(label=label, fair_prob=prob, fair_c=prob * 100.0, **e,
                         bid=(q or {}).get("bid"), mid=(q or {}).get("mid")))
    return rows
