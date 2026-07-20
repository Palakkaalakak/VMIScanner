"""Comparison charts for the multi-vintage VMI backtest (2000 / 2015 / 2020)."""
import json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
curves = pd.read_csv(os.path.join(HERE, "vintage_curves.csv"),
                     index_col=0, parse_dates=True)
res = json.load(open(os.path.join(HERE, "vintage_results.json")))["vintages"]

COL = {"gro": "#d62728", "def": "#1f77b4", "spy": "#7f7f7f"}
LBL = {"gro": "Growth book", "def": "Defensive book", "spy": "SPY"}

# ---- 1. Per-vintage log equity curves (3 panels) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, year in zip(axes, ("2000", "2015", "2020")):
    for k in ("spy", "def", "gro"):
        s = curves[f"{year}_{k}"].dropna()
        ax.plot(s.index, s.values, color=COL[k], lw=1.4, label=LBL[k])
    ax.set_yscale("log")
    ax.set_title(f"{year} vintage — $1M each, log scale", fontsize=12)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    r = res[year]
    txt = (f"Growth  {r['growth']['cagr_pct']:.1f}%/yr\n"
           f"Defens. {r['defensive']['cagr_pct']:.1f}%/yr\n"
           f"SPY     {r['spy']['cagr_pct']:.1f}%/yr")
    ax.text(.98, .02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(fc="white", alpha=.8, ec="gray"))
fig.suptitle("VMI Great-Business books vs SPY — three vintages "
             "(PIT scanner picks, era wide-moat review, anti-bubble slices)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, "vintage_curves.png"), dpi=130)
plt.close(fig)

# ---- 2. Aligned: growth of $1 by years-since-vintage (log) ----
fig, ax = plt.subplots(figsize=(11, 6.5))
styles = {"2000": "-", "2015": "--", "2020": ":"}
for year in ("2000", "2015", "2020"):
    for k in ("gro", "def"):
        s = curves[f"{year}_{k}"].dropna()
        yrs = (s.index - s.index[0]).days / 365.25
        ax.plot(yrs, s.values / 1e6, color=COL[k], ls=styles[year], lw=1.6,
                label=f"{year} {LBL[k]}")
    s = curves[f"{year}_spy"].dropna()
    yrs = (s.index - s.index[0]).days / 365.25
    ax.plot(yrs, s.values / 1e6, color=COL["spy"], ls=styles[year], lw=1.1,
            label=f"{year} SPY", alpha=.8)
ax.set_yscale("log")
ax.set_xlabel("Years since vintage start")
ax.set_ylabel("Growth of $1 (log)")
ax.set_title("Aligned equity curves — all vintages, years since inception")
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "vintage_aligned.png"), dpi=130)
plt.close(fig)

# ---- 3. CAGR + maxDD bars ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
years = ("2000", "2015", "2020")
x = range(len(years)); w = .27
for i, (k, jk) in enumerate((("gro", "growth"), ("def", "defensive"), ("spy", "spy"))):
    cag = [res[y][jk]["cagr_pct"] for y in years]
    dd = [res[y][jk]["max_drawdown_pct"] for y in years]
    a1.bar([p + (i - 1) * w for p in x], cag, w, color=COL[k], label=LBL[k])
    a2.bar([p + (i - 1) * w for p in x], dd, w, color=COL[k], label=LBL[k])
    for p, v in zip(x, cag):
        a1.text(p + (i - 1) * w, v + .3, f"{v:.1f}", ha="center", fontsize=8)
    for p, v in zip(x, dd):
        a2.text(p + (i - 1) * w, v + .5, f"{v:.0f}", ha="center", fontsize=8)
a1.set_xticks(list(x)); a1.set_xticklabels(years); a1.set_ylabel("CAGR %")
a1.set_title("CAGR by vintage"); a1.grid(alpha=.3, axis="y"); a1.legend(fontsize=9)
a2.set_xticks(list(x)); a2.set_xticklabels(years); a2.set_ylabel("Max drawdown %")
a2.set_title("Max drawdown by vintage"); a2.grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "vintage_bars.png"), dpi=130)
plt.close(fig)

print("wrote vintage_curves.png, vintage_aligned.png, vintage_bars.png")
