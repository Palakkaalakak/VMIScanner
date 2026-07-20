"""Expanded chart suite for the VMI 2000-2013 two-account backtest.

Reads: eq_defensive.csv eq_growth.csv eq_spy.csv stats2.json trades.json
       weekly_adj.csv weekly_ohlc.csv
Writes everything into charts/:
  portfolio_<acct>_line.png        individual equity line + correction shading
  portfolio_<acct>_candles.png     monthly OHLC candlestick of equity curve
  portfolio_both_candles.png       both portfolios' candles side by side
  stocks/<TICKER>_<acct>.png       monthly candles + 40wSMA + IV + trade markers
                                   + position-value panel
  grid_<acct>.png                  16-stock normalized price grid
  annual_returns.png               yearly bar chart def vs growth vs SPY
  contrib_<acct>.png               per-stock contribution to final value
  rolling_cagr.png                 rolling 3y CAGR
  drawdown_compare.png             underwater curves, all three
  stock_returns.csv                per-stock return table
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import numpy as np
import pandas as pd

os.makedirs("charts/stocks", exist_ok=True)

INITIAL = 1_000_000.0
eq = {
    "defensive": pd.read_csv("eq_defensive.csv", index_col=0, parse_dates=True).iloc[:, 0],
    "growth": pd.read_csv("eq_growth.csv", index_col=0, parse_dates=True).iloc[:, 0],
    "spy": pd.read_csv("eq_spy.csv", index_col=0, parse_dates=True).iloc[:, 0],
}
stats = json.load(open("stats2.json"))
tj = json.load(open("trades.json"))
px = pd.read_csv("weekly_adj.csv", index_col=0, parse_dates=True)
ohlc = pd.read_csv("weekly_ohlc.csv", header=[0, 1], index_col=0, parse_dates=True)
sma40 = px.rolling(40).mean()
anchor = pd.Timestamp(tj["defensive"]["anchor"])

COL = {"defensive": "#1f77b4", "growth": "#2ca02c", "spy": "#888888"}
NICE = {"defensive": "Defensive account", "growth": "Growth account", "spy": "S&P 500 (SPY)"}


def iv_series(acct, t, idx):
    iv0 = tj[acct]["iv0"].get(t)
    g = tj[acct]["iv_growth"].get(t)
    if iv0 is None:
        return None
    yrs = (idx - anchor).days / 365.25
    return pd.Series(iv0 * (1 + g) ** yrs, index=idx)


def monthly_ohlc_from_series(s):
    return pd.DataFrame({
        "Open": s.resample("ME").first(), "High": s.resample("ME").max(),
        "Low": s.resample("ME").min(), "Close": s.resample("ME").last(),
    }).dropna()


# ---------------------------------------------------------------- 1. lines
for acct in ("defensive", "growth"):
    s = eq[acct]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(s.index, s / 1e6, color=COL[acct], lw=1.8, label=NICE[acct])
    ax.plot(eq["spy"].index, eq["spy"] / 1e6, color="#aaaaaa", lw=1.2,
            label="S&P 500 (SPY)")
    for c in stats[acct]["corrections_7pct_plus"]:
        x0 = pd.Timestamp(c["peak"])
        x1 = pd.Timestamp(c["recovered"]) if c["recovered"] else s.index[-1]
        ax.axvspan(x0, x1, color="red", alpha=0.07)
        tg = pd.Timestamp(c["trough"])
        ax.annotate(f"-{c['depth_pct']}%", (tg, s.loc[:tg].min() / 1e6),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color="darkred")
    st = stats[acct]
    ax.set_title(f"{NICE[acct]} 2000-2013 — +{st['total_return_pct']}% total, "
                 f"{st['cagr_pct']}% CAGR, final ${st['final_value']:,} "
                 f"(red = 7%+ corrections)")
    ax.set_ylabel("Portfolio value ($M)")
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"charts/portfolio_{acct}_line.png", dpi=110)
    plt.close(fig)

# ------------------------------------------------------------ 2. candles
mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit",
                           wick="inherit")
style = mpf.make_mpf_style(base_mpf_style="yahoo", marketcolors=mc,
                           gridstyle=":", gridcolor="#dddddd")
for acct in ("defensive", "growth"):
    mo = monthly_ohlc_from_series(eq[acct]) / 1e6
    spy_mo = eq["spy"].resample("ME").last().reindex(mo.index) / 1e6
    ap = [mpf.make_addplot(spy_mo, color="#888888", width=1.0,
                           label="S&P 500 (SPY)")]
    fig, axes = mpf.plot(mo, type="candle", style=style, addplot=ap,
                         figsize=(14, 7), returnfig=True,
                         ylabel="Portfolio value ($M)",
                         title=f"{NICE[acct]} — monthly candles of portfolio equity")
    axes[0].legend(loc="upper left")
    fig.savefig(f"charts/portfolio_{acct}_candles.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

# both portfolios candles, stacked
fig = plt.figure(figsize=(14, 10))
for i, acct in enumerate(("defensive", "growth")):
    ax = fig.add_subplot(2, 1, i + 1)
    mo = monthly_ohlc_from_series(eq[acct]) / 1e6
    w = 18
    for d, r in mo.iterrows():
        up = r.Close >= r.Open
        c = "#26a69a" if up else "#ef5350"
        ax.plot([d, d], [r.Low, r.High], color=c, lw=0.8, zorder=2)
        ax.add_patch(plt.Rectangle((mdates.date2num(d) - w / 2,
                                    min(r.Open, r.Close)), w,
                                   abs(r.Close - r.Open) or 0.001,
                                   color=c, zorder=3))
    ax.plot(eq["spy"].index, eq["spy"] / 1e6, color="#999999", lw=1.0,
            label="S&P 500 (SPY)")
    ax.set_title(f"{NICE[acct]} — monthly candles")
    ax.set_ylabel("$M"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("charts/portfolio_both_candles.png", dpi=110)
plt.close(fig)

# ------------------------------------------------- 3. per-stock charts
MARKER = {"BUY": ("^", "#00aa00", 110), "ADD": ("^", "#66cc33", 80),
          "BUY_REPL": ("^", "#0066cc", 110), "SELL": ("v", "#cc0000", 120)}
rows = []
for acct in ("defensive", "growth"):
    trades = pd.DataFrame(tj[acct]["trades"])
    trades["date"] = pd.to_datetime(trades["date"])
    for t in sorted(set(trades["ticker"])):
        tt = trades[trades["ticker"] == t]
        wk = ohlc[t][["Open", "High", "Low", "Close"]].loc["2000-01-01":"2013-12-31"].dropna()
        mo = pd.DataFrame({
            "Open": wk["Open"].resample("ME").first(),
            "High": wk["High"].resample("ME").max(),
            "Low": wk["Low"].resample("ME").min(),
            "Close": wk["Close"].resample("ME").last()}).dropna()
        idx = mo.index
        sma_mo = sma40[t].resample("ME").last().reindex(idx)
        ivs = iv_series(acct, t, idx)

        # position value panel: shares step function from trades
        sh = pd.Series(0.0, index=px.index)
        cum = 0.0
        for _, r in tt.sort_values("date").iterrows():
            if r["action"] == "SELL":
                cum = 0.0
            else:
                cum += r["amount"] / r["price"]
            sh.loc[r["date"]:] = cum
        posval = (sh * px[t]).loc["2000-01-01":"2013-12-31"]
        pos_mo = posval.resample("ME").last().reindex(idx) / 1e3

        ap = [mpf.make_addplot(sma_mo, color="#ff9900", width=1.2,
                               label="40-week SMA"),
              mpf.make_addplot(pos_mo, panel=1, color=COL[acct], width=1.2,
                               ylabel="Position ($k)")]
        if ivs is not None:
            ap.insert(1, mpf.make_addplot(ivs, color="#9933cc", width=1.2,
                                          linestyle="--", label="DCF intrinsic value"))
        for act, (m, c, sz) in MARKER.items():
            sel = tt[tt["action"] == act]
            if sel.empty:
                continue
            ser = pd.Series(np.nan, index=idx)
            for _, r in sel.iterrows():
                near = idx[idx.searchsorted(r["date"]).clip(0, len(idx) - 1)]
                ser.loc[near] = r["price"]
            ap.append(mpf.make_addplot(ser, type="scatter", marker=m,
                                       color=c, markersize=sz, label=act))
        tr_str = ", ".join(f"{r['action']}@{r['price']}" for _, r in
                           tt.sort_values("date").iterrows())
        fig, axes = mpf.plot(mo, type="candle", style=style, addplot=ap,
                             figsize=(14, 8), returnfig=True,
                             panel_ratios=(3, 1), volume=False,
                             ylabel="Price (adj $)",
                             title=f"{t} ({NICE[acct]}) — monthly candles, "
                                   f"40w SMA, DCF IV, trades")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.savefig(f"charts/stocks/{t}_{acct}.png", dpi=100,
                    bbox_inches="tight")
        plt.close(fig)

        # return-table row
        buys = tt[tt["action"] != "SELL"]
        invested = buys["amount"].sum()
        fh = tj[acct]["final_holdings"].get(t, {})
        fv = fh.get("final_value", 0)
        p0 = wk["Close"].iloc[0]
        p1 = wk["Close"].iloc[-1]
        rows.append({"account": acct, "ticker": t,
                     "n_trades": len(tt), "invested": invested,
                     "final_value": fv,
                     "gain_pct": round((fv / invested - 1) * 100, 1) if invested and fv else None,
                     "px_2000": round(p0, 2), "px_2013": round(p1, 2),
                     "stock_return_pct": round((p1 / p0 - 1) * 100, 1)})
        print("stock chart:", t, acct)

ret = pd.DataFrame(rows).sort_values(["account", "final_value"],
                                     ascending=[True, False])
ret.to_csv("charts/stock_returns.csv", index=False)

# ------------------------------------------------- 4. normalized grids
BOOKS = {"defensive": list(tj["defensive"]["iv0"]),
         "growth": list(tj["growth"]["iv0"])}
for acct, book in BOOKS.items():
    fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    for ax, t in zip(axs.flat, sorted(book)):
        s = px[t].loc["2000-01-01":"2013-12-31"].dropna()
        ax.plot(s.index, s / s.iloc[0], color=COL[acct], lw=1.0)
        ax.plot(eq["spy"].index, eq["spy"] / INITIAL, color="#bbbbbb", lw=0.8)
        ax.set_title(f"{t}  {s.iloc[-1] / s.iloc[0]:.1f}x", fontsize=10)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    fig.suptitle(f"{NICE[acct]} — every holding, normalized to 1.0 at Jan 2000 "
                 f"(grey = SPY)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"charts/grid_{acct}.png", dpi=100)
    plt.close(fig)

# ------------------------------------------------- 5. annual returns
yr = pd.DataFrame({k: eq[k].resample("YE").last() for k in eq})
yr.loc[pd.Timestamp("1999-12-31")] = INITIAL
yr = yr.sort_index().pct_change().dropna() * 100
yr.index = yr.index.year
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(yr))
for i, k in enumerate(("defensive", "growth", "spy")):
    ax.bar(x + (i - 1) * 0.27, yr[k], 0.27, color=COL[k], label=NICE[k])
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(yr.index)
ax.set_ylabel("Annual return (%)")
ax.set_title("Annual returns — both accounts vs S&P 500")
ax.grid(alpha=0.3, axis="y"); ax.legend()
for i, k in enumerate(("defensive", "growth", "spy")):
    for xi, v in zip(x, yr[k]):
        ax.text(xi + (i - 1) * 0.27, v + (0.8 if v >= 0 else -2.2),
                f"{v:.0f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig("charts/annual_returns.png", dpi=110)
plt.close(fig)

# ------------------------------------------------- 6. contributions
for acct in ("defensive", "growth"):
    fh = tj[acct]["final_holdings"]
    ser = pd.Series({t: v["final_value"] for t, v in fh.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(ser.index, ser / 1e3, color=COL[acct])
    for b, v in zip(bars, ser):
        ax.text(v / 1e3 + 5, b.get_y() + b.get_height() / 2,
                f"${v / 1e3:,.0f}k", va="center", fontsize=8)
    cash = tj[acct]["final_cash"]
    ax.set_xlabel("Final position value ($k)")
    ax.set_title(f"{NICE[acct]} — what each stock is worth at end-2013 "
                 f"(cash left: ${cash / 1e3:,.0f}k)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"charts/contrib_{acct}.png", dpi=110)
    plt.close(fig)

# ------------------------------------------------- 7. rolling 3y CAGR
fig, ax = plt.subplots(figsize=(14, 6))
for k in ("defensive", "growth", "spy"):
    r = (eq[k] / eq[k].shift(156)) ** (1 / 3) - 1
    ax.plot(r.index, r * 100, color=COL[k], lw=1.4, label=NICE[k])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Rolling 3-year CAGR (%)")
ax.set_title("Rolling 3-year CAGR — consistency of compounding")
ax.grid(alpha=0.3); ax.legend()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("charts/rolling_cagr.png", dpi=110)
plt.close(fig)

# ------------------------------------------------- 8. underwater curves
fig, ax = plt.subplots(figsize=(14, 6))
for k in ("defensive", "growth", "spy"):
    dd = (eq[k] / eq[k].cummax() - 1) * 100
    ax.plot(dd.index, dd, color=COL[k], lw=1.2, label=NICE[k])
    ax.fill_between(dd.index, dd, 0, color=COL[k], alpha=0.12)
ax.axhline(-7, color="red", ls=":", lw=1, label="-7% threshold")
ax.set_ylabel("Drawdown from peak (%)")
ax.set_title("Underwater chart — every drawdown, all three portfolios")
ax.grid(alpha=0.3); ax.legend(loc="lower left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("charts/drawdown_compare.png", dpi=110)
plt.close(fig)

print("\nDONE. files:", len(os.listdir("charts")) - 1, "+",
      len(os.listdir("charts/stocks")), "stock charts")
print(ret.to_string(index=False))
