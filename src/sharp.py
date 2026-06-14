"""Sharp no-vig ANCHOR — the discipline rail.

The point: our model has a known draw/underdog lean, so "model disagrees with Kalshi" is NOT enough to
bet. We only trust a signal when an INDEPENDENT sharp price ALSO disagrees with Kalshi. This module
produces that independent fair probability by removing the vig from a sharp book's odds.

Sources, in priority order:
  1. MANUAL  — you type the sharp (e.g. Pinnacle) decimal odds for a match; we de-vig them. Always works,
               no key. Persisted to outputs/anchors.json.
  2. SharpAPI Pinnacle — auto, if you set SHARP_API_KEY (free signup). Best-effort: exact schema is
               confirmed on first real call; until a key exists this is dormant.
Polymarket (no auth) anchors the TOURNAMENT-winner odds only (it has no per-match markets).

De-vig: raw implied prob p_i = 1/decimal_odds_i; the book's overround = sum(p_i) > 1. We strip it to
get fair probs that sum to 1 (proportional, or the more accurate 'power' method for favs/longshots).
"""
import json
import os
import urllib.parse
import urllib.request

from config import OUTPUTS

ANCHORS_PATH = OUTPUTS / "anchors.json"


# ---------------- de-vig math ----------------

def implied_from_decimal(odds):
    return [1.0 / float(o) for o in odds]


def devig(probs, method="power"):
    """Strip the overround from raw implied probabilities -> fair probs summing to 1."""
    probs = [max(1e-9, float(p)) for p in probs]
    s = sum(probs)
    if s <= 0:
        return None
    if method == "proportional" or len(probs) < 2:
        return [p / s for p in probs]
    # power method: find k with sum(p_i**k) == 1 (k>1 for overround>0)
    lo, hi = 0.3, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(p ** k for p in probs) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    out = [p ** k for p in probs]
    t = sum(out)
    return [x / t for x in out]


def fair_from_decimal(odds, method="power"):
    """[home,draw,away] decimal odds -> de-vigged {home,draw,away} fair probabilities."""
    f = devig(implied_from_decimal(odds), method)
    if not f:
        return None
    return {"home": f[0], "draw": f[1], "away": f[2]}


# ---------------- manual anchor store ----------------

def _load_anchors():
    if ANCHORS_PATH.exists():
        try:
            return json.loads(ANCHORS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_anchors(d):
    ANCHORS_PATH.write_text(json.dumps(d, indent=2))


def set_manual(home, away, odds_home, odds_draw, odds_away):
    """Store sharp decimal odds for a fixture (what you read off Pinnacle / oddsportal)."""
    d = _load_anchors()
    d[f"{home}|{away}"] = {"oh": float(odds_home), "od": float(odds_draw), "oa": float(odds_away)}
    _save_anchors(d)
    return fair_from_decimal([odds_home, odds_draw, odds_away])


def clear_manual(home=None, away=None):
    if home is None:
        _save_anchors({})
        return
    d = _load_anchors()
    d.pop(f"{home}|{away}", None)
    _save_anchors(d)


def _manual_fair(home, away):
    d = _load_anchors().get(f"{home}|{away}")
    if not d:
        return None
    return fair_from_decimal([d["oh"], d["od"], d["oa"]])


# ---------------- SharpAPI / Pinnacle (keyed, best-effort) ----------------

def _key():
    return os.environ.get("SHARP_API_KEY")


def _pinnacle_fair(home, away):
    """Best-effort SharpAPI Pinnacle no-vig. Dormant until SHARP_API_KEY is set AND the endpoint/schema
    is confirmed on a first real call. Returns {home,draw,away} or None."""
    key = _key()
    if not key:
        return None
    base = os.environ.get("SHARP_API_URL", "https://api.sharpapi.io/v1/odds/soccer")
    try:
        qs = urllib.parse.urlencode({"home": home, "away": away, "book": "pinnacle", "novig": "true"})
        req = urllib.request.Request(f"{base}?{qs}", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        # Try common shapes; adjust once the real schema is in hand.
        if "novig" in data:
            n = data["novig"]
            return {"home": n["home"], "draw": n["draw"], "away": n["away"]}
        if "odds" in data:
            o = data["odds"]
            return fair_from_decimal([o["home"], o["draw"], o["away"]])
    except Exception:
        return None
    return None


def anchor_fair(home, away):
    """Independent sharp fair probs for a match: {source, home, draw, away} or None.
    Manual overrides take priority (you looked it up), else SharpAPI Pinnacle."""
    m = _manual_fair(home, away)
    if m:
        return {"source": "manual", **m}
    p = _pinnacle_fair(home, away)
    if p:
        return {"source": "pinnacle", **p}
    return None


def available():
    return bool(_key()) or bool(_load_anchors())


# ---------------- Polymarket tournament-winner anchor (no auth) ----------------

def polymarket_champion():
    """{team: fair_prob} for the World Cup winner from Polymarket (de-vigged across all teams). {} on error."""
    try:
        url = "https://gamma-api.polymarket.com/markets?closed=false&limit=500"
        with urllib.request.urlopen(url, timeout=8) as resp:
            ms = json.loads(resp.read().decode())
        raw = {}
        for m in (ms if isinstance(ms, list) else ms.get("data", [])):
            q = str(m.get("question", ""))
            if "win the 2026 FIFA World Cup" not in q:
                continue
            team = q.replace("Will ", "").split(" win the")[0].strip()
            price = None
            op = m.get("outcomePrices")
            if isinstance(op, str):
                try:
                    op = json.loads(op)
                except Exception:
                    op = None
            if isinstance(op, list) and op:
                price = float(op[0])           # P(Yes)
            if price is not None:
                raw[team] = price
        tot = sum(raw.values())
        return {t: p / tot for t, p in raw.items()} if tot > 0 else {}
    except Exception:
        return {}


if __name__ == "__main__":
    print("de-vig demo: Pinnacle-style odds Brazil 2.10 / draw 3.30 / Morocco 3.90")
    print("  proportional:", {k: round(v, 3) for k, v in fair_from_decimal([2.10, 3.30, 3.90], "proportional").items()})
    print("  power       :", {k: round(v, 3) for k, v in fair_from_decimal([2.10, 3.30, 3.90], "power").items()})
    print("\nPolymarket WC champion (top 6):")
    pc = polymarket_champion()
    for t, p in sorted(pc.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {t:14s} {p*100:5.1f}%")
