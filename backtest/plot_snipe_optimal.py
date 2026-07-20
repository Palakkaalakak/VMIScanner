"""Chart the fine roof sweep: the corrections-caught vs profit-target cliff."""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import pandas as pd

d = json.load(open("snipe_optimal.json"))
COL = {"defensive": "#1f77b4", "growth": "#2ca02c"}
LS = {"off10": "-", "off5": "--"}

fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
for ax, acct in zip(axes, ("growth", "defensive")):
    for off in ("off10", "off5"):
        sw = pd.DataFrame(d[acct][off]["sweep"])
        c = COL[acct] if off == "off10" else "#ff9900"
        ax.plot(sw["roof_pct"], sw["final"] / 1e6, LS[off], color=c, lw=1.5,
                label=f"bottom+{off[3:]}% entry")
        b = d[acct][off]["best_roof_only"]
        ax.plot(b["roof_pct"], b["final"] / 1e6, "*", color=c, ms=20, zorder=5)
        ax.annotate(f"{b['roof_pct']}%\n${b['final']/1e6:.0f}M",
                    (b["roof_pct"], b["final"] / 1e6),
                    xytext=(8, 6), textcoords="offset points", fontsize=9,
                    fontweight="bold", color=c)
    ax.axvline(200, color="red", ls=":", lw=1, label="+200% (original)")
    ax.set_ylabel("Final pot ($M)")
    ax.set_title(f"{acct.title()}: compounded $100k final vs profit target "
                 f"(no time limit) — cliffs = losing a correction to a "
                 f"too-greedy roof")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
axes[1].set_xlabel("Profit target (%)")
fig.suptitle("The roof/corrections-caught tradeoff — sawtooth rises while legs "
             "stay caught, cliffs when a leg gets trapped", y=0.995)
fig.tight_layout()
fig.savefig("charts/snipe_optimal_sweep.png", dpi=110)
plt.close(fig)
print("saved")
