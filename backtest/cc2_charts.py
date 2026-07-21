"""Charts for CC v2 (IV-calibrated, PDF roll mechanics)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "cc2_results.json")))
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]
CFGS = ["OLD_g25_lowbeta", "g25_lowbeta|sma", "g25_lowbeta|always",
        "g25_lowbeta|always_d45", "g25_lowbeta|always_h5",
        "g25_lowbeta|always_h10", "g<=25|sma", "lowbeta_half|always",
        "all|sma", "all|always", "all|always_h5", "all|always_h10"]

mat = np.array([[r[y][c]["cagr_pct"] - r[y]["none"]["cagr_pct"]
                 for y in YEARS] for c in CFGS])

fig, ax = plt.subplots(figsize=(11, 6.5))
im = ax.imshow(mat, cmap="RdYlGn", vmin=-3, vmax=13, aspect="auto")
ax.set_xticks(range(len(YEARS)), YEARS)
ax.set_yticks(range(len(CFGS)), CFGS, fontsize=8)
for i in range(len(CFGS)):
    for j in range(len(YEARS)):
        ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center",
                fontsize=7.5)
ax.set_title("CC v2: CAGR uplift vs no-CC baseline (pts/yr)\n"
             "IV-calibrated premiums + PDF roll mechanics; "
             "h5/h10 = 5%/10% bid-ask haircut")
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cc2_heatmap.png"), dpi=110)

# old vs new model on the ideal config
fig2, ax2 = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(YEARS))
old = [r[y]["OLD_g25_lowbeta"]["cagr_pct"] - r[y]["none"]["cagr_pct"]
       for y in YEARS]
new_sma = [r[y]["g25_lowbeta|sma"]["cagr_pct"] - r[y]["none"]["cagr_pct"]
           for y in YEARS]
new_alw = [r[y]["g25_lowbeta|always"]["cagr_pct"] - r[y]["none"]["cagr_pct"]
           for y in YEARS]
new_h5 = [r[y]["g25_lowbeta|always_h5"]["cagr_pct"]
          - r[y]["none"]["cagr_pct"] for y in YEARS]
ax2.bar(x - .3, old, .2, label="v1 model (RV, settle+wait)", color="#999")
ax2.bar(x - .1, new_sma, .2, label="v2, SMA gate", color="#4c9")
ax2.bar(x + .1, new_alw, .2, label="v2, always covered", color="#27c")
ax2.bar(x + .3, new_h5, .2, label="v2, always + 5% haircut",
        color="#a6f")
ax2.set_xticks(x, YEARS)
ax2.axhline(0, color="k", lw=.5)
ax2.set_ylabel("CAGR uplift (pts/yr)")
ax2.set_title("g25_lowbeta covered calls: old vs improved model")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(HERE, "cc2_oldnew.png"), dpi=110)
print("charts saved")
