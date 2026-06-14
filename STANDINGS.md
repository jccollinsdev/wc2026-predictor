# 🏆 WC2026 Model — Live Standings

_Auto-generated 2026-06-13 21:06 · predictions, scoring, and a Kelly-sized mock portfolio. Not financial advice._

## 📊 Model efficacy (1X2 predictions vs actual)

| Matches settled | Accuracy | Mean RPS | Coin-flip RPS |
|---|---|---|---|
| 2 / 62 | 0.0% | 0.2165 | 0.2222 |

_RPS is the proper score for ordered W/D/L outcomes — **lower is better**, and beating the 0.2222 coin-flip baseline means the model carries real signal._

## 💵 Kelly $1000 portfolio (risk-sized, fee-aware)

- **Equity: $982.00**  (start $1000, return **-1.8%**)
- Cash $809.59 · at risk $172.41 · max drawdown -1.8%
- Positions: 0 settled, 12 open · win rate —

_¼-Kelly sizing, capped at 10% of bankroll per line; stakes compound with the live bankroll._

## 🧾 Flat paper book ($10/bet — clean signal test)

- Settled 0 · realized **$+0.00** · ROI — · win — · **CLV —** · fees $6.73

_CLV (closing-line value) is the truest edge signal: consistently positive CLV means the model beats the market's closing price, regardless of any single result._

## ✅ Recent settled predictions

| Date | Match | Model pick | Actual | ✓ | RPS |
|---|---|---|---|---|---|
| 2026-06-13 | Brazil v Morocco | home | draw | ❌ | 0.115 |
| 2026-06-13 | Qatar v Switzerland | away | draw | ❌ | 0.318 |

## 📈 Open Kelly positions

| Match | Bet | Entry | Contracts | Stake | ¼-Kelly |
|---|---|---|---|---|---|
| Haiti v Scotland | Scotland win | 61¢ | 86 | $52.46 | 5.3% |
| Netherlands v Japan | Draw | 27¢ | 40 | $10.80 | 1.1% |
| Sweden v Tunisia | Tunisia win | 22¢ | 50 | $11.00 | 1.1% |
| Saudi Arabia v Uruguay | Draw | 21¢ | 111 | $23.31 | 2.4% |
| Argentina v Algeria | Draw | 21¢ | 66 | $13.86 | 1.4% |
| Argentina v Algeria | Algeria win | 11¢ | 44 | $4.84 | 0.5% |
| Austria v Jordan | Draw | 18¢ | 68 | $12.24 | 1.2% |
| Austria v Jordan | Jordan win | 11¢ | 44 | $4.84 | 0.5% |
| France v Senegal | Draw | 22¢ | 56 | $12.32 | 1.3% |
| France v Senegal | Senegal win | 13¢ | 106 | $13.78 | 1.4% |
| Iraq v Norway | Iraq win | 6¢ | 127 | $7.62 | 0.8% |
| Iraq v Norway | Draw | 13¢ | 111 | $14.43 | 1.5% |

---

Model = time-decayed Dixon–Coles + World-Football Elo + XGBoost stacker + feature-aware score engine; signals are disciplined by a sharp no-vig anchor and net of Kalshi fees. Regenerate with `python src/track.py`. **Paper money only — not financial advice.**