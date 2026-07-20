"""Charts for the daily-rebalance leverage study."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

res = json.load(open("leverage_results.json"))
curves = pd.read_csv("leverage_curves.csv", index_col=0, parse_dates=True)

COL = {"defensive": "#1f77b4", "growth": "#2ca02c", "spy": "#888888"}
NICE = {"defensive": "Defensive", "growth": "Growth", "spy": "SPY"}

# ---- 1. CAGR vs leverage frontier + max drawdown ------------------------
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
for acct in ("defensive", "growth", "spy"):
    rows = res[acct]
    lev = [r["leverage"] for r in rows]
    cagr = [r["cagr_pct"] for r in rows]
    dd = [r["max_drawdown_pct"] for r in rows]
    ok = [not r["disqualified"] for r in rows]
    ax.plot(lev, cagr, "-o", color=COL[acct], label=NICE[acct])
    ax2.plot(lev, dd, "-o", color=COL[acct], label=NICE[acct])
    for x, y, q in zip(lev, cagr, ok):
        if not q:
            ax.plot(x, y, "x", color="red", markersize=14, markeredgewidth=3)
    best = res.get(acct + "_best")
    if best:
        ax.annotate(f"best {best['leverage']}x\n{best['cagr_pct']}%",
                    (best["leverage"], best["cagr_pct"]),
                    xytext=(10, 12), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=COL[acct])
        ax.plot(best["leverage"], best["cagr_pct"], "*", color=COL[acct],
                markersize=22, zorder=5)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("CAGR 2000-2013 (%)")
ax.set_title("Daily-rebalanced leverage (UPRO-style): CAGR vs leverage factor\n"
             "red ✗ = disqualified (>99% drawdown / blown) · ★ = best qualifying")
ax.grid(alpha=0.3); ax.legend()
ax2.axhline(99, color="red", ls="--", lw=1.2, label="99% = disqualification")
ax2.set_ylabel("Max drawdown (%)"); ax2.set_xlabel("Leverage (x)")
ax2.grid(alpha=0.3); ax2.legend()
ax2.set_xticks([r["leverage"] for r in res["defensive"]])
fig.tight_layout()
fig.savefig("charts/leverage_frontier.png", dpi=110)
plt.close(fig)

# ---- 2. levered equity curves (log scale) -------------------------------
for acct in ("defensive", "growth"):
    fig, ax = plt.subplots(figsize=(14, 7))
    for Y in (1.0, 2.0, 3.0, 4.0, 5.0):
        s = curves[f"{acct}_{Y}"]
        best = res[acct + "_best"]["leverage"]
        lw = 2.4 if Y == best else 1.2
        ax.plot(s.index, s / 1e6, lw=lw,
                label=f"{Y:.0f}x" + (" ← best" if Y == best else ""))
    ax.plot(curves["spy_1.0"].index, curves["spy_1.0"] / 1e6, color="#999999",
            lw=1.0, ls=":", label="SPY 1x")
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value ($M, log scale)")
    ax.set_title(f"{NICE[acct]} account with daily-rebalanced leverage "
                 f"(financing at T-bill + 0.95% ER)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"charts/leverage_curves_{acct}.png", dpi=110)
    plt.close(fig)

# ---- 3. SPY carnage for contrast ----------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for Y in (1.0, 2.0, 3.0, 4.0, 5.0):
    ax.plot(curves[f"spy_{Y}"].index, curves[f"spy_{Y}"] / 1e6,
            lw=1.4, label=f"SPY {Y:.0f}x")
ax.set_yscale("log")
ax.set_ylabel("Value ($M, log)")
ax.set_title("Why leverage needs a great portfolio: leveraged SPY 2000-2013 "
             "was destroyed by volatility drag through two 50%+ crashes")
ax.grid(alpha=0.3, which="both"); ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("charts/leverage_spy.png", dpi=110)
plt.close(fig)
print("saved 4 leverage charts")
