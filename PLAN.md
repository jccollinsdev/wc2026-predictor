# World Cup 2026 Match Predictor — Quant Plan

Predict, for any WC2026 match: full-time score distribution, 1X2 outcome, and who leads at
half-time. CPU-only (Mac M1). Built on free historical data.

## Outputs
1. Full-time: most-likely scoreline, full P(x,y) grid, expected goals per side, top-5 scorelines.
2. Outcome: calibrated P(home win), P(draw), P(away win).
3. First half: P(home leads at HT), P(level), P(away leads), most-likely HT score, expected 1H goals.

## Architecture — 4-component ensemble
- **Dixon–Coles** (time-decayed bivariate Poisson, custom MLE + analytic gradient): score engine.
  Produces attack/defense per team, home effect, low-score rho correction, full scoreline grid.
- **World-Football Elo** (computed in-house over all matches): strongest single predictor.
  K by importance (WC=60, qualifiers=40, continental finals=50, friendly=20), G margin multiplier,
  +100 home only when neutral=FALSE, shootout counts as draw.
- **XGBoost stacker**: calibrates final 1X2 from DC probs + Elo + form/rest/venue features.
- **Poisson thinning** (f=0.4474, measured from goalscorers.csv): first-half = Poisson(f*lambda).

Skipped: LSTM/Transformer/LLM (no gain, infeasible on CPU); PyMC hierarchical as live core.

## Data (martj42 international_results, CC0, GitHub raw — no Kaggle needed)
- results.csv (49,478 matches 1872-2026; already includes WC2026 fixtures as unplayed rows)
- goalscorers.csv (has `minute` -> native first-half goals)
- shootouts.csv (Elo shootout handling)
Optional lift (Kaggle): EA FC player ratings + Wikipedia squads -> squad strength + injuries.

## Measured base rates (from the data, sanity anchors)
- Outcomes: home 49.0% / draw 22.7% / away 28.3%. Avg goals: home 1.76, away 1.18.
- Home advantage: non-neutral home-win 50.7% vs neutral 44.2%.
- First-half goal fraction f = 0.4474 (47,606 goals). Goal rate rises through the match.

## Features (stacker)
Ratings (Elo diff, Elo win-prob), DC-implied 1X2, DC lambdas, rolling form (N=5 goals for/against,
points), rest days + diff, venue (is_neutral/home), tournament tier, recent H2H.

## Validation
Walk-forward, strictly time-based. Refit DC per cutoff; Elo pre-match by construction; stacker on
past only. Metrics: RPS (primary), log-loss, 1X2 accuracy, exact-score hit-rate, HT-lead accuracy.
Targets: RPS 0.17-0.19, accuracy ~55-60%, exact-score ~10-13%, reproduce HT split ~44/28/16.

## Build order
1. data.py: load/clean (coerce '00'/'NA', dates, played flag, tournament tiers).
2. elo.py: World-Football Elo over full history; store pre-match ratings per match.
3. dixon_coles.py: time-decayed DC MLE with analytic gradient; grid -> scoreline/xG/1X2.
4. first_half.py: thin lambdas by f -> HT grid -> lead probs.
5. features.py: engineer leakage-free feature table.
6. stacker.py: XGBoost 1X2 meta-learner + blend with DC + calibration.
7. backtest.py: walk-forward metrics.
8. predict.py: unified API + CLI -> full prediction for any fixture.
Target: verify on USA vs Paraguay (2026-06-12, Group D, USA home).
