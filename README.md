# ⚽ WC2026 Predictor & Live Value Tracker

A quant-grade World Cup 2026 match-prediction model built **only from free, public data** (no paid
APIs) — plus a localhost app that prices every betting market live, compares the model's fair value to
**Kalshi**, disciplines its signals with a **sharp no-vig anchor**, and tracks its own efficacy and a
**Kelly-sized $1,000 mock portfolio** in public.

> 📈 **[See the live standings → STANDINGS.md](STANDINGS.md)** — model accuracy, RPS, paper P&L, and the Kelly portfolio, updated as matches settle.
>
> ⚠️ **Paper money only. Not financial advice.** This is a research project; markets are hard and the model can be wrong (and has been — see the standings).

---

## What it does

- **Predicts every WC2026 match** — full-time score distribution, 1X2 (win/draw/win), who leads at
  half-time, and a full board of markets (totals, BTTS, first-to-score, correct score, etc.).
- **Live in-play re-pricing** — given the current score + minute, every market re-prices on the
  Poisson clock (e.g. at 1–0 in the 88th minute, "win by 2+" decays toward zero automatically).
- **Edge detection vs Kalshi** — pulls live Kalshi prices (no auth) and shows the model's fair value,
  a recommended entry price, and the net-of-fees edge for each outcome.
- **Sharp discipline** — a no-vig anchor (manual Pinnacle odds, or SharpAPI) vetoes signals where an
  independent sharp price agrees with the market. Only signals confirmed by *both* fire as a GO.
- **Tracks itself honestly** — model accuracy + RPS, a flat paper book (with CLV), and a Kelly-sized
  $1,000 portfolio, all settled against real Kalshi results and published in `STANDINGS.md`.

## The model

A four-part ensemble, all leakage-free / walk-forward validated:

1. **Time-decayed Dixon–Coles** bivariate Poisson (custom MLE + analytic gradient) — the score engine.
2. **World-Football Elo** (importance-weighted K, margin-of-victory, neutral-venue aware) — strength.
3. **XGBoost stacker** — calibrates 1X2 from DC probs + Elo + form + squad strength + novel features.
4. **Feature-aware score engine** — XGBoost Poisson predicting full- and first-half goal rates.

Squad strength comes from EA FC + multi-year FIFA ratings aggregated by nationality, with an injury
overlay applied at prediction time. On the **real 2018 & 2022 World Cups it beats the bookmakers'
accuracy** and sits within ~0.002–0.01 RPS of closing odds.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py            # pull latest results (no key needed)

# one-off predictions
python src/predict.py "Brazil" "Morocco" --neutral --date 2026-06-13
python src/predict.py --all-wc2026      # every fixture -> outputs/wc2026_predictions.csv

# the live app (localhost, auto-refresh every 60s)
python src/server.py --port 8000        # http://127.0.0.1:8000

# refresh the public tracker (predictions + Kelly portfolio + STANDINGS.md)
python src/track.py
```

### Optional live feeds (free, optional keys)

| Feed | Used for | Key |
|---|---|---|
| **Kalshi** | live market prices + settlement | none (public) |
| **api-football** | confirmed lineups + live in-play scores | free key → `export API_FOOTBALL_KEY=…` (or a gitignored `.env`) |
| **SharpAPI / Pinnacle** | automatic sharp no-vig anchor | free key → `export SHARP_API_KEY=…` (or type Pinnacle odds manually in the app) |

## How the tracker works

`python src/track.py` builds the model once, logs new predictions, settles finished ones against the
real Kalshi result, opens fresh Kelly-sized positions, and regenerates `STANDINGS.md`. Commit & push
to publish:

```bash
python src/track.py && git add -A && git commit -m "update standings" && git push
```

Key metrics:
- **RPS** — proper score for ordered W/D/L; **lower is better**, beating 0.2222 (coin-flip) = real signal.
- **CLV (closing-line value)** — did entries beat the market's closing price? Positive CLV over many
  bets = genuine edge, independent of any single result.
- **Kelly portfolio** — ¼-Kelly sizing, capped at 10%/line, fees modeled; equity compounds.

## Data sources (all free / no-auth)

- **Matches/goals/shootouts** — [martj42/international_results](https://github.com/martj42/international_results) (CC0)
- **Player ratings** — EA FC 26 + multi-year FIFA mirrors, aggregated by nationality
- **Market prices** — [Kalshi public API](https://docs.kalshi.com/) (World Cup series `KXWCGAME`, etc.)
- **Odds/results history** — football-data.co.uk closing odds (validation)

## Disclaimer

Research / educational project. **Paper money only — nothing here is financial advice.** Prediction
markets are efficient and the model is frequently wrong; the published standings exist precisely to be
honest about that. Never bet money you can't afford to lose.

🤖 Built with [Claude Code](https://claude.com/claude-code).
