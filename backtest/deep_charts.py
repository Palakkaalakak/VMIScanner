"""Charts for the 7-vintage all-growth backtest (1990-2020)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
curves = pd.read_csv(os.path.join(HERE, "deep_curves.csv"),
                     index_col=0, parse_dates=True)
res = json.load(open(os.path.join(HERE, "deep_results.json")))
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]

# ---- 1. grid of per-vintage log curves ----
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for ax, y in zip(axes.flat, YEARS):
    g = curves[f"{y}_gro"].dropna()
    b = curves[f"{y}_bench"].dropna()
    ax.plot(b.index, b.values, color="#7f7f7f", lw=1.1,
            label=res[y]["bench_ticker"])
    ax.plot(g.index, g.values, color="#d62728", lw=1.4, label="Growth book")
    ax.set_yscale("log")
    ax.set_title(f"{y} vintage", fontsize=12)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    r = res[y]
    ax.text(.98, .02,
            f"book {r['growth']['cagr_pct']:.1f}%/yr\n"
            f"bench {r['benchmark']['cagr_pct']:.1f}%/yr",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            family="monospace", bbox=dict(fc="white", alpha=.8, ec="gray"))
axes.flat[-1].axis("off")
fig.suptitle("VMI all-growth books vs benchmark — seven vintages, $1M each, "
             "log scale (16 stocks, 25% sector cap, anti-bubble slices)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "deep_grid.png"), dpi=125)
plt.close(fig)

# ---- 2. aligned growth-of-$1 (log) ----
cmap = plt.cm.viridis
fig, ax = plt.subplots(figsize=(12, 7))
for i, y in enumerate(YEARS):
    g = curves[f"{y}_gro"].dropna()
    yrs = (g.index - g.index[0]).days / 365.25
    ax.plot(yrs, g.values / 1e6, color=cmap(i / 6), lw=1.7,
            label=f"{y} book ({res[y]['growth']['cagr_pct']:.1f}%)")
    b = curves[f"{y}_bench"].dropna()
    yrs = (b.index - b.index[0]).days / 365.25
    ax.plot(yrs, b.values / 1e6, color=cmap(i / 6), lw=.9, ls=":", alpha=.7)
ax.set_yscale("log")
ax.set_xlabel("Years since vintage start")
ax.set_ylabel("Growth of $1 (log)")
ax.set_title("Aligned equity curves — solid = VMI growth book, "
             "dotted = benchmark")
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "deep_aligned.png"), dpi=130)
plt.close(fig)

# ---- 3. CAGR + maxDD + excess bars ----
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
x = range(len(YEARS)); w = .38
gc = [res[y]["growth"]["cagr_pct"] for y in YEARS]
bc = [res[y]["benchmark"]["cagr_pct"] for y in YEARS]
gd = [res[y]["growth"]["max_drawdown_pct"] for y in YEARS]
bd = [res[y]["benchmark"]["max_drawdown_pct"] for y in YEARS]
a1.bar([p - w / 2 for p in x], gc, w, color="#d62728", label="Growth book")
a1.bar([p + w / 2 for p in x], bc, w, color="#7f7f7f", label="Benchmark")
for p, v in zip(x, gc):
    a1.text(p - w / 2, v + .2, f"{v:.1f}", ha="center", fontsize=8)
for p, v in zip(x, bc):
    a1.text(p + w / 2, v + .2, f"{v:.1f}", ha="center", fontsize=8)
a1.set_xticks(list(x)); a1.set_xticklabels(YEARS)
a1.set_ylabel("CAGR %"); a1.set_title("CAGR by vintage")
a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=9)
a2.bar([p - w / 2 for p in x], gd, w, color="#d62728", label="Growth book")
a2.bar([p + w / 2 for p in x], bd, w, color="#7f7f7f", label="Benchmark")
a2.set_xticks(list(x)); a2.set_xticklabels(YEARS)
a2.set_ylabel("Max drawdown %"); a2.set_title("Max drawdown by vintage")
a2.grid(alpha=.3, axis="y")
ex = [g - b for g, b in zip(gc, bc)]
a3.bar(x, ex, .55, color=["#2ca02c" if e > 0 else "#d62728" for e in ex])
for p, v in zip(x, ex):
    a3.text(p, v + (.1 if v > 0 else -.35), f"{v:+.1f}", ha="center",
            fontsize=9)
a3.set_xticks(list(x)); a3.set_xticklabels(YEARS)
a3.set_ylabel("Excess CAGR vs benchmark (pts/yr)")
a3.set_title("Alpha by vintage"); a3.grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "deep_bars.png"), dpi=130)
plt.close(fig)
print("wrote deep_grid.png, deep_aligned.png, deep_bars.png")
