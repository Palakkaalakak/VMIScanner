"""Charts for the covered-call config grid."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "cc_results.json")))
YEARS = list(res.keys())
CFGS = ["none", "g<=8", "g<=10", "g<=12", "g<=15", "g<=20", "g<=25",
        "g<=30", "ex_top4_growth", "ex_top2_growth", "lowbeta_half", "all"]

# ---- heatmap of CAGR uplift vs none ----
up = np.array([[res[y][c]["cagr_pct"] - res[y]["none"]["cagr_pct"]
                for c in CFGS] for y in YEARS])
fig, ax = plt.subplots(figsize=(13, 5.5))
vmax = max(abs(up.min()), abs(up.max()))
im = ax.imshow(up, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(CFGS)))
ax.set_xticklabels(CFGS, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(YEARS)))
ax.set_yticklabels(YEARS)
for i in range(len(YEARS)):
    for j in range(len(CFGS)):
        ax.text(j, i, f"{up[i, j]:+.1f}", ha="center", va="center",
                fontsize=8)
ax.set_title("Covered-call overlay: CAGR uplift vs no-CC baseline "
             "(pts/yr) by config and vintage")
fig.colorbar(im, ax=ax, shrink=.8, label="pts/yr")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cc_heatmap.png"), dpi=130)
plt.close(fig)

# ---- avg uplift + worst case per config ----
avg = up.mean(axis=0)
worst = up.min(axis=0)
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(CFGS))
ax.bar(x - .2, avg, .4, color="#2ca02c", label="avg uplift (7 vintages)")
ax.bar(x + .2, worst, .4, color="#d62728", label="worst vintage")
for p, v in zip(x, avg):
    ax.text(p - .2, v + .04, f"{v:+.2f}", ha="center", fontsize=8)
for p, v in zip(x, worst):
    ax.text(p + .2, v - .12, f"{v:+.2f}", ha="center", fontsize=8)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(x); ax.set_xticklabels(CFGS, rotation=35, ha="right",
                                     fontsize=9)
ax.set_ylabel("CAGR uplift vs none (pts/yr)")
ax.set_title("Which stocks to sell covered calls on — average gain vs "
             "worst-case cost")
ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "cc_config.png"), dpi=130)
plt.close(fig)
print("wrote cc_heatmap.png, cc_config.png")
