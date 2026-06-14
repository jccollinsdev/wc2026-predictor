"""One-shot updater for the public tracker. Builds the model ONCE, then:
  1. snapshots new 1X2 predictions for upcoming matches
  2. settles any finished predictions / paper bets / portfolio positions against real Kalshi results
  3. opens fresh Kelly-sized positions in the $1000 portfolio (and flat paper bets)
  4. regenerates STANDINGS.md

Run it whenever you want to refresh the public scorecard (e.g. daily), then commit & push:
    python src/track.py && git add -A && git commit -m "update standings" && git push
"""
import argparse


def update(days=4, flat_paper=True, tentative=True):
    from predict import Predictor
    import predictions
    import portfolio
    import paper
    import standings

    print("[track] building model (one-time) ...")
    pred = Predictor(verbose=False)

    added = predictions.snapshot(days=14, pred=pred)
    s_pred = predictions.settle()
    print(f"[track] predictions: +{added} logged, {s_pred} newly settled")

    opened = portfolio.update(days=days, pred=pred, tentative=tentative)
    print(f"[track] portfolio: opened {len(opened)} Kelly position(s)")

    if flat_paper:
        logged, blocked = paper.auto(days=days, tentative=tentative, pred=pred)
        print(f"[track] paper: +{len(logged)} flat bet(s), {blocked} blocked by sharp anchor")

    path = standings.write()
    print(f"[track] wrote {path}")

    pr = predictions.report(); po = portfolio.report(); pa = paper.report()
    print("\n========================  SUMMARY  ========================")
    if pr["accuracy"] is not None:
        print(f"  model: {pr['settled']}/{pr['total']} settled | acc {pr['accuracy']*100:.1f}% | "
              f"RPS {pr['mean_rps']:.4f} (coin-flip {pr['baseline_rps']:.4f})")
    else:
        print(f"  model: {pr['total']} predictions logged, none settled yet")
    print(f"  Kelly $1000 -> ${po['equity']:.2f} ({po['ret']*100:+.1f}%) | "
          f"{po['n_open']} open, {po['n_settled']} settled")
    clv = "n/a" if pa["avg_clv"] is None else f"{pa['avg_clv']:+.1f}c"
    print(f"  flat paper: {pa['settled']} settled, ${pa['pnl']:+.2f}, CLV {clv}")
    print("===========================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--no-flat", action="store_true", help="skip the flat paper book")
    ap.add_argument("--confirmed-only", action="store_true", help="only sharp-confirmed GO (no tentative)")
    args = ap.parse_args()
    update(days=args.days, flat_paper=not args.no_flat, tentative=not args.confirmed_only)


if __name__ == "__main__":
    main()
