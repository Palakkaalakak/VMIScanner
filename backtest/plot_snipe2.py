"""Charts for correction-sniping v2 (no-time-limit + sweeps)."""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = json.load(open("correction_snipe2.json"))
COL = {"defensive": "#1f77b4", "growth": "#2ca02c"}
LS = {"off10": "-", "off5": "--"}

# ---- 1. days-to-TP200 per trade (no limit) ------------------------------
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
for ax, acct in zip(axs, ("growth", "defensive")):
    for off, shift in (("off10", -0.2), ("off5", 0.2)):
        tr = d[acct][off]["user_tp200_nolimit"]["trades"]
        x = np.arange(len(tr)) + shift
        days = [t["days_held"] for t in tr]
        c = COL[acct] if off == "off10" else "#ff9900"
        ax.bar(x, days, 0.38, color=c,
               label=f"entry bottom+{off[3:]}%")
        for xi, t in zip(x, tr):
            ax.text(xi, t["days_held"] + 30, f"dip\n{t['min_mult']:.2f}",
                    ha="center", fontsize=6.5, color="#555")
    labels = [t["trough"] for t in d[acct]["off10"]["user_tp200_nolimit"]["trades"]]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(365, color="red", ls=":", lw=1, label="1 year")
    ax.set_ylabel("Days to hit +200%")
    ax.set_title(f"{acct.title()}: every snipe DID reach +200% with no time "
                 f"limit — but some took 6-7 years and dipped >90% first")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("charts/snipe2_days_to_tp.png", dpi=110)
plt.close(fig)

# ---- 2. time-limit sweep at TP200 ---------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for acct in ("growth", "defensive"):
    for off in ("off10", "off5"):
        sw = pd.DataFrame(d[acct][off]["tlimit_sweep_tp200"])
        x = [tl if tl else 5110 for tl in sw["max_days"]]
        a1.plot(x, sw["total_profit_fresh"] / 1e6, LS[off] + "o", ms=4,
                color=COL[acct], label=f"{acct} +{off[3:]}%")
        a2.plot(x, sw["compounded_final"] / 1e6, LS[off] + "o", ms=4,
                color=COL[acct], label=f"{acct} +{off[3:]}%")
for ax, ttl in ((a1, "Fresh $100k per trade — total profit"),
                (a2, "Compounded $100k (overlap-aware)")):
    ax.set_xscale("log")
    ax.axvline(5110, color="gray", ls=":", lw=1)
    ax.text(5110, ax.get_ylim()[1]*0.02, " NO LIMIT", fontsize=8, color="gray")
    ax.set_xlabel("Max hold (days, log)"); ax.set_ylabel("$M")
    ax.set_title(ttl); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.suptitle("Time-limit sweep at TP +200%: longer is better for fresh money; "
             "compounded peaks near 120-150d (quality) or no-limit", y=1.02)
fig.tight_layout()
fig.savefig("charts/snipe2_tlimit_sweep.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- 3. roof sweep, no limit --------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
for acct in ("growth", "defensive"):
    for off in ("off10", "off5"):
        sw = pd.DataFrame(d[acct][off]["roof_sweep_nolimit"])
        a1.plot(sw["roof_pct"], sw["total_profit_fresh"] / 1e6, LS[off],
                color=COL[acct], label=f"{acct} +{off[3:]}%")
        a2.plot(sw["roof_pct"], sw["compounded_final"] / 1e6, LS[off],
                color=COL[acct], label=f"{acct} +{off[3:]}%")
a1.set_title("Fresh money: higher roof = more (never blows up)")
a2.set_title("Compounded: optimum ~375-650% roof")
for ax in (a1, a2):
    ax.axvline(200, color="red", ls=":", lw=1, label="+200% (base)")
    ax.set_xlabel("Profit roof (%)"); ax.set_ylabel("$M")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Roof sweep with NO time limit", y=1.02)
fig.tight_layout()
fig.savefig("charts/snipe2_roof_sweep.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved snipe2 charts")
