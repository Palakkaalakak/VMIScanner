"""Charts: no-sell dividend-reinvest run + leverage frontier per vintage."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
curves = pd.read_csv(os.path.join(HERE, "div_curves.csv"),
                     index_col=0, parse_dates=True)
res = json.load(open(os.path.join(HERE, "div_results.json")))
old = json.load(open(os.path.join(HERE, "deep_results.json")))
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]

# ---- 1. grid: div-reinvest book vs benchmark ----
fig, axes = plt.subplots(2, 4, figsize=(22, 9))
for ax, y in zip(axes.flat, YEARS):
    g = curves[f"{y}_gro"].dropna()
    b = curves[f"{y}_bench"].dropna()
    ax.plot(b.index, b.values, color="#7f7f7f", lw=1.1,
            label=res[y]["bench_ticker"])
    ax.plot(g.index, g.values, color="#9467bd", lw=1.4,
            label="Book (no sells, div reinvest)")
    ax.set_yscale("log"); ax.grid(alpha=.3, which="both")
    ax.set_title(f"{y} vintage", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    r = res[y]
    ax.text(.98, .02,
            f"book  {r['growth']['cagr_pct']:.1f}%/yr\n"
            f"bench {r['benchmark']['cagr_pct']:.1f}%/yr\n"
            f"divs  ${r['dividends_collected']/1e6:.1f}M",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            family="monospace", bbox=dict(fc="white", alpha=.8, ec="gray"))
axes.flat[-1].axis("off")
fig.suptitle("No scandal sells + disciplined dividend reinvestment — "
             "seven vintages, $1M each (log scale)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "div_grid.png"), dpi=125)
plt.close(fig)

# ---- 2. leverage frontier: CAGR vs L per vintage ----
fig, ax = plt.subplots(figsize=(11, 6.5))
cmap = plt.cm.viridis
for i, y in enumerate(YEARS):
    lev = json.load(open(os.path.join(HERE, f"leverage_{y}.json")))
    grid = [(L, c) for L, c in lev["grid"] if c is not None]
    Ls = [g[0] for g in grid]; cs = [g[1] for g in grid]
    ax.plot(Ls, cs, color=cmap(i / 6), lw=1.6,
            label=f"{y}  (L*={lev['ideal_L_cagr_max']:.2f}, "
                  f"Kelly={lev['kelly_full']:.1f})")
    ax.scatter([lev["ideal_L_cagr_max"]], [lev["cagr_at_ideal_pct"]],
               color=cmap(i / 6), zorder=5, s=28)
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.axvline(1.5, color="black", ls="--", lw=1, alpha=.6)
ax.text(1.52, ax.get_ylim()[0] + 2, "1.5x practical", fontsize=8, rotation=90)
ax.set_xlabel("Leverage L (borrow at era 10Y path)")
ax.set_ylabel("CAGR % of levered book")
ax.set_title("Leverage frontier by vintage — CAGR-maximizing L "
             "(dots) vs practical range")
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "div_leverage.png"), dpi=130)
plt.close(fig)

# ---- 3. old (scandal-sell, adj prices) vs new (no-sell + divs) CAGR ----
fig, ax = plt.subplots(figsize=(11, 5.5))
x = range(len(YEARS)); w = .38
oc = [old[y]["growth"]["cagr_pct"] for y in YEARS]
nc = [res[y]["growth"]["cagr_pct"] for y in YEARS]
ax.bar([p - w / 2 for p in x], oc, w, color="#d62728",
       label="Scandal-sell run")
ax.bar([p + w / 2 for p in x], nc, w, color="#9467bd",
       label="No sells + div reinvest")
for p, v in zip(x, oc):
    ax.text(p - w / 2, v + .15, f"{v:.1f}", ha="center", fontsize=8)
for p, v in zip(x, nc):
    ax.text(p + w / 2, v + .15, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(YEARS)
ax.set_ylabel("CAGR %")
ax.set_title("Effect of holding through scandals + disciplined dividend "
             "reinvestment")
ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "div_vs_old.png"), dpi=130)
plt.close(fig)
print("wrote div_grid.png, div_leverage.png, div_vs_old.png")
