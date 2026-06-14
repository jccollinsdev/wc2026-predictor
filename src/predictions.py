"""Model-efficacy tracker: log every 1X2 prediction, settle it against the real result, score it.

This is the public scorecard of how good the MODEL is (independent of betting): for each match we
record the predicted win/draw/win probabilities; once the match settles on Kalshi we record the actual
outcome and compute whether the pick was right and the Ranked Probability Score (RPS, the proper
ordinal metric for 1X2). Aggregate accuracy + mean RPS vs a coin-flip baseline = the efficacy number.

State: outputs/predictions_log.json (persistent, committed so people can track it over time).
"""
import argparse
import json
from datetime import datetime

from config import OUTPUTS
import kalshi

LOG_PATH = OUTPUTS / "predictions_log.json"


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _load():
    if LOG_PATH.exists():
        try:
            return json.loads(LOG_PATH.read_text())
        except Exception:
            pass
    return {"preds": []}


def _save(d):
    LOG_PATH.write_text(json.dumps(d, indent=2))


def _rps(probs, actual_idx):
    """Ranked Probability Score for ordered outcomes [home, draw, away]; 0 = perfect, lower better."""
    outcome = [0, 0, 0]
    outcome[actual_idx] = 1
    cum_p = cum_o = 0.0
    s = 0.0
    for i in range(3):
        cum_p += probs[i]
        cum_o += outcome[i]
        s += (cum_p - cum_o) ** 2
    return s / 2.0


def snapshot(days=14, pred=None):
    """Record predictions for upcoming matches not already logged. Returns count added."""
    import pandas as pd
    from data import upcoming_matches
    if pred is None:
        from predict import Predictor
        pred = Predictor(verbose=False)
    up = upcoming_matches(pred.df).sort_values("date")
    lo = pred.asof.normalize(); hi = lo + pd.Timedelta(days=days)
    fx = up[(up["date"] >= lo) & (up["date"] < hi)]
    d = _load()
    seen = {(x["date"], x["home"], x["away"]) for x in d["preds"]}
    added = 0
    for _, m in fx.iterrows():
        key = (str(m["date"].date()), m["home_team"], m["away_team"])
        if key in seen:
            continue
        r = pred.predict_match(m["home_team"], m["away_team"], neutral=bool(m["neutral"]), date=m["date"])
        mk = kalshi.find_match(r["home"], r["away"]) or {}
        d["preds"].append(dict(
            logged=_now(), date=key[0], home=r["home"], away=r["away"],
            p_home=round(r["p_home"], 4), p_draw=round(r["p_draw"], 4), p_away=round(r["p_away"], 4),
            pick=r["outcome"], proj_score=f"{r['headline_score'][0]}-{r['headline_score'][1]}",
            tickers={s: mk.get(s) for s in ("home", "draw", "away")},
            status="open", actual=None, correct=None, rps=None))
        added += 1
    _save(d)
    return added


def settle():
    """Resolve any open predictions whose match has settled on Kalshi. Returns count settled."""
    d = _load()
    idx = {"home": 0, "draw": 1, "away": 2}
    n = 0
    for x in d["preds"]:
        if x["status"] != "open":
            continue
        actual = None
        for side in ("home", "draw", "away"):
            tk = (x.get("tickers") or {}).get(side)
            if not tk:
                continue
            info = kalshi.market_result(tk)
            if info and info.get("settled") and info["result"] == "yes":
                actual = side
                break
        if actual is None:
            continue
        x["actual"] = actual
        x["correct"] = int(x["pick"] == actual)
        x["rps"] = round(_rps([x["p_home"], x["p_draw"], x["p_away"]], idx[actual]), 4)
        x["status"] = "settled"
        n += 1
    if n:
        _save(d)
    return n


def report():
    d = _load()
    settled = [x for x in d["preds"] if x["status"] == "settled"]
    acc = (sum(x["correct"] for x in settled) / len(settled)) if settled else None
    rps = (sum(x["rps"] for x in settled) / len(settled)) if settled else None
    return dict(total=len(d["preds"]), settled=len(settled), open=len(d["preds"]) - len(settled),
                accuracy=acc, mean_rps=rps, baseline_rps=0.2222)   # coin-flip 1/3 each -> RPS 0.222


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    ps = sub.add_parser("snapshot"); ps.add_argument("--days", type=int, default=14)
    sub.add_parser("settle"); sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "snapshot":
        print(f"logged {snapshot(days=args.days)} new prediction(s).")
    elif args.cmd == "settle":
        print(f"settled {settle()} prediction(s).")
    rp = report()
    print(f"\nMODEL EFFICACY: {rp['settled']}/{rp['total']} settled", end="")
    if rp["accuracy"] is not None:
        print(f" | accuracy {rp['accuracy']*100:.1f}% | RPS {rp['mean_rps']:.4f} "
              f"(coin-flip {rp['baseline_rps']:.4f})")
    else:
        print(" | (no settled matches yet)")


if __name__ == "__main__":
    main()
