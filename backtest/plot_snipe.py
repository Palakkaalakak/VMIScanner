"""Charts for the correction-sniping 5x study."""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = json.load(open("correction_snipe.json"))
COL = {"defensive": "#1f77b4", "growth": "#2ca02c"}

# 1. per-trade P/L bars (roof +200%)
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
for ax, acct in zip(axs, ("growth", "defensive")):
    tr = d[acct]["roof_200"]["trades"]
    labels = [t["entry"] for t in tr]
    pl = [t["profit_on_100k"] / 1e3 for t in tr]
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in pl]
    ax.bar(range(len(pl)), pl, color=colors)
    for i, (v, t) in enumerate(zip(pl, tr)):
        ax.text(i, v + (1.5 if v >= 0 else -4), f"{v:+.0f}k",
                ha="center", fontsize=8)
        ax.text(i, min(pl) - 12, f"dip {t['min_mult_during']:.2f}x",
                ha="center", fontsize=7, color="#666")
    ax.set_xticks(range(len(pl)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    tot = sum(t["profit_on_100k"] for t in tr)
    ax.set_title(f"{acct.title()} 5x snipes ($100k each, exit at +200% or 3.5mo) "
                 f"— total ${tot:,.0f} · 0 roof hits · 0 blowups")
    ax.set_ylabel("P/L ($k)")
    ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig("charts/snipe_trades.png", dpi=110)
plt.close(fig)

# 2. roof sweep
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for acct in ("growth", "defensive"):
    sw = pd.DataFrame(d[acct]["sweep"])
    a1.plot(sw["roof_pct"], sw["total_profit"] / 1e3, "-o", ms=3,
            color=COL[acct], label=acct.title())
    a2.plot(sw["roof_pct"], sw["compounded"] / 1e3, "-o", ms=3,
            color=COL[acct], label=acct.title())
    bf = d[acct]["ideal_roof_fresh"]; bc = d[acct]["ideal_roof_compounded"]
    a1.plot(bf["roof_pct"], bf["total_profit"] / 1e3, "*", ms=20, color=COL[acct])
    a2.plot(bc["roof_pct"], bc["compounded"] / 1e3, "*", ms=20, color=COL[acct])
for ax, ttl, yl in ((a1, "Total profit (fresh $100k per trade)", "Total P/L ($k)"),
                    (a2, "Compounded (roll proceeds into next trade)", "Final value of $100k ($k)")):
    ax.axvline(200, color="red", ls=":", lw=1, label="+200% roof (asked)")
    ax.set_xlabel("Profit roof (%)"); ax.set_ylabel(yl)
    ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend()
fig.suptitle("Ideal profit roof sweep — ★ = optimum; above ~+125% the roof "
             "never gets hit inside 3.5 months, so curves flatten", y=1.02)
fig.tight_layout()
fig.savefig("charts/snipe_roof_sweep.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved snipe charts")
