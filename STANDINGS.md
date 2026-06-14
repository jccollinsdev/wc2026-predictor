# 🏆 WC2026 Model — Live Standings

_Auto-generated 2026-06-14 06:06 · predictions, scoring, and a Kelly-sized mock portfolio. Not financial advice._

## 📊 Model efficacy (1X2 predictions vs actual)

| Matches settled | Accuracy | Mean RPS | Coin-flip RPS |
|---|---|---|---|
| 3 / 68 | 33.3% | 0.1627 | 0.2222 |

_RPS is the proper score for ordered W/D/L outcomes — **lower is better**, and beating the 0.2222 coin-flip baseline means the model carries real signal._

## 💵 Kelly $1000 portfolio (risk-sized, fee-aware)

- **Equity: $977.95**  (start $1000, return **-2.2%**)
- Cash $801.88 · at risk $176.07 · max drawdown -2.2%
- Positions: 0 settled, 16 open · win rate —

_¼-Kelly sizing, capped at 10% of bankroll per line; stakes compound with the live bankroll._

## 🧾 Flat paper book ($10/bet — clean signal test)

- Settled 2 · realized **$+13.40** · ROI 66.9% · win 100% · **CLV —** · fees $10.88

_CLV (closing-line value) is the truest edge signal: consistently positive CLV means the model beats the market's closing price, regardless of any single result._

## ✅ Recent settled predictions

| Date | Match | Model pick | Actual | ✓ | RPS |
|---|---|---|---|---|---|
| 2026-06-13 | Haiti v Scotland | away | away | ✅ | 0.055 |
| 2026-06-13 | Brazil v Morocco | home | draw | ❌ | 0.115 |
| 2026-06-13 | Qatar v Switzerland | away | draw | ❌ | 0.318 |

## 📈 Open Kelly positions

| Match | Bet | Entry | Contracts | Stake | ¼-Kelly |
|---|---|---|---|---|---|
| Sweden v Tunisia | Tunisia win | 22¢ | 59 | $12.98 | 1.3% |
| Belgium v Egypt | Draw | 24¢ | 32 | $7.68 | 0.8% |
| Saudi Arabia v Uruguay | Draw | 22¢ | 94 | $20.68 | 2.1% |
| Argentina v Algeria | Draw | 21¢ | 66 | $13.86 | 1.4% |
| Argentina v Algeria | Algeria win | 11¢ | 44 | $4.84 | 0.5% |
| Austria v Jordan | Draw | 18¢ | 70 | $12.60 | 1.3% |
| Austria v Jordan | Jordan win | 11¢ | 54 | $5.94 | 0.6% |
| France v Senegal | Draw | 22¢ | 58 | $12.76 | 1.3% |
| France v Senegal | Senegal win | 13¢ | 111 | $14.43 | 1.5% |
| Iraq v Norway | Iraq win | 6¢ | 177 | $10.62 | 1.1% |
| Iraq v Norway | Draw | 13¢ | 135 | $17.55 | 1.8% |
| Ghana v Panama | Panama win | 29¢ | 45 | $13.05 | 1.4% |
| England v Croatia | Draw | 25¢ | 52 | $13.00 | 1.3% |
| Uzbekistan v Colombia | Uzbekistan win | 10¢ | 29 | $2.90 | 0.3% |
| Uzbekistan v Colombia | Draw | 20¢ | 85 | $17.00 | 1.7% |

---

Model = time-decayed Dixon–Coles + World-Football Elo + XGBoost stacker + feature-aware score engine; signals are disciplined by a sharp no-vig anchor and net of Kalshi fees. Regenerate with `python src/track.py`. **Paper money only — not financial advice.**