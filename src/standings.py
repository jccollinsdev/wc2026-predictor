"""Generate STANDINGS.md — the public scorecard people can follow in the GitHub repo.

Combines three persistent records into one markdown report:
  * model efficacy  (predictions.py)  — accuracy + RPS on settled matches
  * flat paper book (paper.py)        — did the signals win? + CLV
  * Kelly $1000 portfolio (portfolio.py) — risk-sized equity curve

Run via track.py (which refreshes everything first), or standalone to re-render from current state.
"""
from datetime import datetime

from config import OUTPUTS
import paper
import portfolio
import predictions

REPO_ROOT = OUTPUTS.parent


def _pct(x, dp=1):
    return "—" if x is None else f"{x*100:.{dp}f}%"


def build_md():
    pr = predictions.report()
    pa = paper.report()
    po = portfolio.report()
    preds = predictions._load()["preds"]
    settled = [x for x in preds if x["status"] == "settled"]
    L = []
    L.append("# 🏆 WC2026 Model — Live Standings\n")
    L.append(f"_Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
             "predictions, scoring, and a Kelly-sized mock portfolio. Not financial advice._\n")

    L.append("## 📊 Model efficacy (1X2 predictions vs actual)\n")
    L.append("| Matches settled | Accuracy | Mean RPS | Coin-flip RPS |")
    L.append("|---|---|---|---|")
    rps = "—" if pr["mean_rps"] is None else f"{pr['mean_rps']:.4f}"
    L.append(f"| {pr['settled']} / {pr['total']} | {_pct(pr['accuracy'])} | {rps} | "
             f"{pr['baseline_rps']:.4f} |")
    L.append("\n_RPS is the proper score for ordered W/D/L outcomes — **lower is better**, and beating "
             "the 0.2222 coin-flip baseline means the model carries real signal._\n")

    L.append("## 💵 Kelly $1000 portfolio (risk-sized, fee-aware)\n")
    L.append(f"- **Equity: ${po['equity']:.2f}**  (start ${po['start']:.0f}, "
             f"return **{_pct(po['ret'])}**)")
    L.append(f"- Cash ${po['cash']:.2f} · at risk ${po['exposure']:.2f} · "
             f"max drawdown {_pct(po['max_dd'])}")
    L.append(f"- Positions: {po['n_settled']} settled, {po['n_open']} open · "
             f"win rate {_pct(po['win_rate'], 0)}")
    L.append("\n_¼-Kelly sizing, capped at 10% of bankroll per line; stakes compound with the live "
             "bankroll._\n")

    L.append("## 🧾 Flat paper book ($10/bet — clean signal test)\n")
    roi = _pct(pa["roi"])
    clv = "—" if pa["avg_clv"] is None else f"{pa['avg_clv']:+.1f}¢"
    L.append(f"- Settled {pa['settled']} · realized **${pa['pnl']:+.2f}** · ROI {roi} · "
             f"win {_pct(pa['win_rate'], 0)} · **CLV {clv}** · fees ${pa['fees']:.2f}")
    L.append("\n_CLV (closing-line value) is the truest edge signal: consistently positive CLV means "
             "the model beats the market's closing price, regardless of any single result._\n")

    if settled:
        L.append("## ✅ Recent settled predictions\n")
        L.append("| Date | Match | Model pick | Actual | ✓ | RPS |")
        L.append("|---|---|---|---|---|---|")
        for x in settled[-12:][::-1]:
            mark = "✅" if x["correct"] else "❌"
            L.append(f"| {x['date']} | {x['home']} v {x['away']} | {x['pick']} | "
                     f"{x['actual']} | {mark} | {x['rps']:.3f} |")
        L.append("")

    opens = [x for x in portfolio._load()["positions"] if x["status"] == "open"]
    if opens:
        L.append("## 📈 Open Kelly positions\n")
        L.append("| Match | Bet | Entry | Contracts | Stake | ¼-Kelly |")
        L.append("|---|---|---|---|---|---|")
        for x in opens[-15:]:
            L.append(f"| {x['home']} v {x['away']} | {x['label']} | {x['entry_c']}¢ | "
                     f"{x['contracts']} | ${x['stake']:.2f} | {x['kelly_f']*100:.1f}% |")
        L.append("")

    L.append("---\n")
    L.append("Model = time-decayed Dixon–Coles + World-Football Elo + XGBoost stacker + feature-aware "
             "score engine; signals are disciplined by a sharp no-vig anchor and net of Kalshi fees. "
             "Regenerate with `python src/track.py`. **Paper money only — not financial advice.**")
    return "\n".join(L)


def write(path=None):
    out = path or (REPO_ROOT / "STANDINGS.md")
    out.write_text(build_md())
    return out


if __name__ == "__main__":
    paper.settle(); predictions.settle()
    p = portfolio._load(); portfolio.settle(p); portfolio._save(p)
    path = write()
    print(f"wrote {path}")
