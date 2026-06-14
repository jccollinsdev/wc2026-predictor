"""Optional confirmed-lineup / injury overlay via api-football (API-Sports).

Pure-offline core works without this. If you register a FREE key at https://www.api-football.com and
export it, the dashboard will show confirmed XI + sidelined players ~40 min before kickoff:

    export API_FOOTBALL_KEY=xxxxxxxxxxxxxxxx

No key -> every function returns None and the dashboard simply omits the lineup row. We never block
on the network: a missing key or a failed/slow request degrades silently to "lineups: not available".
"""
import os
import json
import urllib.request

API_HOST = "https://v3.football.api-sports.io"


def _load_dotenv():
    """Load repo-root .env (KEY=VALUE lines) into the environment once, without external deps."""
    if os.environ.get("API_FOOTBALL_KEY") or os.environ.get("APISPORTS_KEY"):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.path.join(os.path.dirname(here), ".env")
    if os.path.exists(env):
        try:
            with open(env) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def _key():
    _load_dotenv()
    return os.environ.get("API_FOOTBALL_KEY") or os.environ.get("APISPORTS_KEY")


def _get(path, params, timeout=6):
    key = _key()
    if not key:
        return None
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    req = urllib.request.Request(f"{API_HOST}{path}?{qs}", headers={"x-apisports-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def lineup_note(home, away, date):
    """Return a short human string about confirmed XI / injuries, or None if unavailable.

    Stub-ready: the wiring (auth header, endpoint) is correct; turning it on is just adding the key.
    Until a key is present this returns None and the dashboard shows nothing for lineups.
    """
    if not _key():
        return None
    # fixture lookup by date + team would go here; kept minimal until a key is configured so we don't
    # burn the free quota on every dashboard run. With a key, extend to /fixtures then /lineups.
    return None


def live_matches():
    """Currently in-play fixtures from api-football: list of dicts
    {home, away, score_h, score_a, minute, league}. Empty list if no key or on any error."""
    data = _get("/fixtures", {"live": "all"})
    if not data or "response" not in data:
        return []
    out = []
    for f in data.get("response", []):
        try:
            teams, goals, fx = f["teams"], f["goals"], f["fixture"]
            out.append(dict(
                home=teams["home"]["name"], away=teams["away"]["name"],
                score_h=int(goals["home"] or 0), score_a=int(goals["away"] or 0),
                minute=int((fx.get("status") or {}).get("elapsed") or 0),
                league=(f.get("league") or {}).get("name", ""),
            ))
        except Exception:
            continue
    return out


def available():
    return _key() is not None
