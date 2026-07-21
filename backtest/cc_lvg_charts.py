"""Charts for the CC x leverage x Sharpe study."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "cc_lvg_results.json")))
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]
CFGS = ["base", "cc_ideal", "cc_ideal_h5", "cc_all", "cc_all_h5"]
LBL = {"base": "no CC", "cc_ideal": "CC ideal (g25lb)",
       "cc_ideal_h5": "CC ideal +5% cost", "cc_all": "CC all",
       "cc_all_h5": "CC all +5% cost"}
COL = {"base": "#888", "cc_ideal": "#27c", "cc_ideal_h5": "#7ad",
       "cc_all": "#c72", "cc_all_h5": "#eb9"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
x = np.arange(len(YEARS))
w = 0.16
for k, c in enumerate(CFGS):
    sh = [r[y][c]["unlevered"]["sharpe"] for y in YEARS]
    axes[0].bar(x + (k - 2) * w, sh, w, label=LBL[c], color=COL[c])
    cg = [r[y][c]["unlevered"]["cagr"] for y in YEARS]
    axes[1].bar(x + (k - 2) * w, cg, w, label=LBL[c], color=COL[c])
    lv = [r[y][c]["L_dd55"]["cagr"] for y in YEARS]
    axes[2].bar(x + (k - 2) * w, lv, w, label=LBL[c], color=COL[c])
for ax, t in zip(axes, ["Sharpe (unlevered)", "CAGR % (unlevered)",
                        "CAGR % levered to maxDD<=55%"]):
    ax.set_xticks(x, YEARS, fontsize=8)
    ax.set_title(t, fontsize=10)
    ax.axhline(0, color="k", lw=.5)
axes[0].legend(fontsize=7)
fig.suptitle("Covered calls x leverage: Sharpe & CAGR by vintage")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cc_lvg_summary.png"), dpi=110)

# averages table printed for the wrap-up
print(f"{'config':<16} {'avg CAGR':>9} {'avg Sharpe':>10} "
      f"{'avg L(dd55)':>11} {'avg CAGR@L':>10} {'worst dd@L':>10}")
for c in CFGS:
    cg = np.mean([r[y][c]["unlevered"]["cagr"] for y in YEARS])
    sh = np.mean([r[y][c]["unlevered"]["sharpe"] for y in YEARS])
    lv = np.mean([r[y][c]["L_dd55"]["L"] for y in YEARS])
    lc = np.mean([r[y][c]["L_dd55"]["cagr"] for y in YEARS])
    wd = min([r[y][c]["L_dd55"]["dd"] for y in YEARS])
    print(f"{c:<16} {cg:>8.2f}% {sh:>10.3f} {lv:>11.2f} {lc:>9.2f}% "
          f"{wd:>9.1f}%")
print("charts saved")
