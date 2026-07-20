"""Chart: both VMI accounts vs S&P 500 (2000-2013), corrections marked."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

eqA = pd.read_csv("eq_defensive.csv", index_col=0, parse_dates=True).iloc[:, 0]
eqB = pd.read_csv("eq_growth.csv", index_col=0, parse_dates=True).iloc[:, 0]
spy = pd.read_csv("eq_spy.csv", index_col=0, parse_dates=True).iloc[:, 0]
S = json.load(open("stats2.json"))

fig, (ax, axd) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2]})

ax.plot(eqB.index, eqB / 1e6, lw=2.0, color="#2ca02c",
        label=f"Account B — GROWTH (CAGR {S['growth']['cagr_pct']}%, "
              f"total +{S['growth']['total_return_pct']}%)")
ax.plot(eqA.index, eqA / 1e6, lw=2.0, color="#1f77b4",
        label=f"Account A — DEFENSIVE (CAGR {S['defensive']['cagr_pct']}%, "
              f"total +{S['defensive']['total_return_pct']}%)")
ax.plot(spy.index, spy / 1e6, lw=1.5, color="#888888",
        label=f"S&P 500 incl. dividends (CAGR {S['spy']['cagr_pct']}%, "
              f"total +{S['spy']['total_return_pct']}%)")

for dd in S["defensive"]["corrections_7pct_plus"]:
    tr = pd.Timestamp(dd["trough"])
    ax.annotate(f"-{dd['depth_pct']}%", xy=(tr, eqA.loc[tr] / 1e6),
                xytext=(0, -18), textcoords="offset points", ha="center",
                fontsize=8, color="#1f77b4", fontweight="bold")
for dd in S["growth"]["corrections_7pct_plus"]:
    if dd["depth_pct"] >= 10:  # annotate only the big ones to avoid clutter
        tr = pd.Timestamp(dd["trough"])
        ax.annotate(f"-{dd['depth_pct']}%", xy=(tr, eqB.loc[tr] / 1e6),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=8, color="#2ca02c", fontweight="bold")

for d, txt, c in [("2002-07-12", "BMY out (accounting fraud)", "#1f77b4"),
                  ("2006-10-20", "UNH out (backdating)", "#1f77b4"),
                  ("2004-07-16", "CAH out (SEC probe)", "#2ca02c")]:
    dt = pd.Timestamp(d)
    ax.axvline(dt, color=c, lw=0.9, ls="--", alpha=0.55)
    ax.annotate(txt, xy=(dt, ax.get_ylim()[1] * 0.98), xytext=(4, -4),
                textcoords="offset points", fontsize=7.5, color=c, va="top",
                rotation=90)

ax.set_title("VMI 'Great Business' Portfolios, Jan 2000 → Dec 2013 — no dot-com exposure\n"
             "16 wide-moat stocks each · DCF-gated entries (buy only under intrinsic value) · "
             "tranche adds at 40-week-SMA support · sell only on fraud/scandal",
             fontsize=12)
ax.set_ylabel("Account value ($ millions; start = $1.0M)")
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

for ser, color, lbl in ((eqB, "#2ca02c", "Growth"), (eqA, "#1f77b4", "Defensive")):
    dd = (ser / ser.cummax() - 1) * 100
    axd.plot(dd.index, dd, color=color, lw=1.1, label=f"{lbl} drawdown")
sdd = (spy / spy.cummax() - 1) * 100
axd.plot(sdd.index, sdd, color="#888888", lw=1.0, label="S&P 500 drawdown")
axd.axhline(-7, color="#d62728", lw=0.8, ls=":", label="-7% correction threshold")
axd.set_ylabel("Drawdown %")
axd.legend(loc="lower left", fontsize=8, ncol=2)
axd.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("vmi_2000_2013_two_accounts.png", dpi=150)
print("saved vmi_2000_2013_two_accounts.png")
