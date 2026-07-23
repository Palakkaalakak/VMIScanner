"""Backtest results tab for the VMI scanner dashboard.

Renders the 2000-2013 two-account VMI backtest (backtest/ folder) as an
interactive dashboard: equity curves, drawdowns, annual returns, per-stock
holdings/trades, correction tables, chart browser and methodology.
Read-only: touches nothing in the scanner pipeline.
"""
import json
import os

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT = os.path.join(REPO_ROOT, "backtest")
CHARTS = os.path.join(BT, "charts")

NICE = {"defensive": "🛡️ Defensive", "growth": "🚀 Growth", "spy": "S&P 500 (SPY)",
        "defensive_cc": "🛡️ Defensive + covered calls",
        "growth_cc": "🚀 Growth + covered calls"}


@st.cache_data(show_spinner=False)
def _load():
    eq = {k: pd.read_csv(os.path.join(BT, f"eq_{k}.csv"), index_col=0,
                         parse_dates=True).iloc[:, 0]
          for k in ("defensive", "growth", "spy")}
    for k in ("defensive", "growth"):
        p = os.path.join(BT, f"eq_{k}_cc.csv")
        if os.path.exists(p):
            eq[k + "_cc"] = pd.read_csv(p, index_col=0,
                                        parse_dates=True).iloc[:, 0]
    stats = json.load(open(os.path.join(BT, "stats2.json")))
    trades = json.load(open(os.path.join(BT, "trades.json")))
    cc_path = os.path.join(BT, "stats_cc_2000_2013.json")
    cc_stats = json.load(open(cc_path)) if os.path.exists(cc_path) else None
    ret_path = os.path.join(CHARTS, "stock_returns.csv")
    stock_ret = pd.read_csv(ret_path) if os.path.exists(ret_path) else None
    return eq, stats, trades, stock_ret, cc_stats


def _img(name, caption=None, subdir=None):
    p = os.path.join(CHARTS, subdir, name) if subdir else os.path.join(CHARTS, name)
    if os.path.exists(p):
        st.image(p, caption=caption, use_container_width=True)
    else:
        st.caption(f"(chart not found: {name})")


def _corrections_df(stats, acct):
    rows = stats[acct]["corrections_7pct_plus"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "peak": "Peak", "trough": "Trough", "recovered": "Recovered",
        "depth_pct": "Depth %", "decline_days": "Decline (days)",
        "recovery_days_from_trough": "Recovery (days)",
        "total_underwater_days": "Underwater (days)"})
    df["Depth %"] = -df["Depth %"]
    return df


def render():
    if not os.path.exists(os.path.join(BT, "stats2.json")):
        st.info("No backtest artifacts found in `backtest/` — run "
                "`python backtest/simulate2.py` then `python backtest/plot_suite.py`.")
        return

    eq, stats, trades, stock_ret, cc_stats = _load()

    st.markdown("### 🕰️ VMI 2000–2013 backtest — standing in January 2000")
    st.caption(
        "Two $1M accounts, 16 wide-moat non-dotcom businesses each, "
        "**DCF-gated entries only** (course-style 20y DCF, rf 6.5%, β×4% MRP), "
        "tranche adds only at the 40-week SMA while under IV, "
        "sells **only** on fraud/scandal (BMY→PFE, UNH→KO, CAH→GPC). "
        "The single allowed piece of hindsight: a defensive sector tilt for "
        "the lost decade. Dividends reinvested via adjusted prices.")

    # ---------- headline metrics ----------
    cols = st.columns(3)
    for col, k in zip(cols, ("growth", "defensive", "spy")):
        s = stats[k]
        col.metric(NICE[k],
                   f"${s['final_value']:,.0f}",
                   f"+{s['total_return_pct']}% · {s['cagr_pct']}% CAGR",
                   delta_color="normal" if k != "spy" else "off")
    g_mult = stats["growth"]["final_value"] / stats["spy"]["final_value"]
    d_mult = stats["defensive"]["final_value"] / stats["spy"]["final_value"]
    st.caption(f"After 14 years the growth book ends **{g_mult:.1f}×** the "
               f"index outcome, the defensive book **{d_mult:.1f}×** — through "
               "the dotcom crash AND the GFC, without a single valuation-"
               "driven sell.")

    sub = st.tabs(["📈 Equity curves", "🕯️ Candles", "📊 Year by year",
                   "🧱 Holdings & contribution", "🔍 Stock charts",
                   "📜 Trade log", "🌊 Corrections", "🧪 Method & caveats"])

    # ---------------- 1. equity curves (interactive) ----------------
    with sub[0]:
        c1, c2 = st.columns([1, 1])
        log_scale = c1.toggle("Log scale", value=False,
                              help="Log scale shows compounding consistency — "
                                   "straight line = constant CAGR.")
        norm = c2.toggle("Normalize to 1.0", value=False)
        df = pd.DataFrame({NICE[k]: eq[k] for k in ("defensive", "growth", "spy")})
        if norm:
            df = df / df.iloc[0]
        try:
            import altair as alt
            long = df.reset_index().melt("index", var_name="Portfolio",
                                         value_name="Value")
            long = long.rename(columns={"index": "Date"})
            y = alt.Y("Value:Q",
                      scale=alt.Scale(type="log") if log_scale else alt.Scale(),
                      title="Portfolio value" + ("" if norm else " ($)"))
            ch = (alt.Chart(long).mark_line().encode(
                x="Date:T", y=y,
                color=alt.Color("Portfolio:N",
                                scale=alt.Scale(domain=list(df.columns),
                                                range=["#1f77b4", "#2ca02c",
                                                       "#888888"])),
                tooltip=["Date:T", "Portfolio:N",
                         alt.Tooltip("Value:Q", format=",.0f")])
                .properties(height=430).interactive())
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.line_chart(df, height=430)
        st.caption("Hover for values · drag to zoom · double-click to reset.")
        with st.expander("Static charts with correction shading"):
            _img("portfolio_growth_line.png")
            _img("portfolio_defensive_line.png")

    # ---------------- 2. candles ----------------
    with sub[1]:
        st.caption("Portfolio equity resampled into **monthly OHLC candles** — "
                   "green/red months show the path, not just the endpoint.")
        _img("portfolio_growth_candles.png")
        _img("portfolio_defensive_candles.png")

    # ---------------- 3. annual returns ----------------
    with sub[2]:
        yr = pd.DataFrame({k: eq[k].resample("YE").last() for k in eq})
        yr.loc[pd.Timestamp("1999-12-31")] = 1_000_000.0
        yr = (yr.sort_index().pct_change().dropna() * 100).round(1)
        yr.index = yr.index.year
        yr.columns = [NICE[k] for k in yr.columns]
        wins_g = int((yr["🚀 Growth"] > yr["S&P 500 (SPY)"]).sum())
        wins_d = int((yr["🛡️ Defensive"] > yr["S&P 500 (SPY)"]).sum())
        neg_g = int((yr["🚀 Growth"] < 0).sum())
        neg_d = int((yr["🛡️ Defensive"] < 0).sum())
        neg_s = int((yr["S&P 500 (SPY)"] < 0).sum())
        a, b, c = st.columns(3)
        a.metric("Years growth beat SPY", f"{wins_g} / {len(yr)}")
        b.metric("Years defensive beat SPY", f"{wins_d} / {len(yr)}")
        c.metric("Negative years (G / D / SPY)", f"{neg_g} / {neg_d} / {neg_s}")
        st.bar_chart(yr, height=340)
        st.dataframe(
            yr.style.format("{:+.1f}%")
              .map(lambda v: "color:#c62828" if v < 0 else "color:#2e7d32"),
            use_container_width=True)
        _img("annual_returns.png", "Same data, labeled bar chart")
        _img("rolling_cagr.png",
             "Rolling 3-year CAGR — the growth account never had a negative "
             "3-year stretch after 2003")

    # ---------------- 4. holdings & contribution ----------------
    with sub[3]:
        acct = st.radio("Account", ["growth", "defensive"], horizontal=True,
                        format_func=lambda k: NICE[k], key="bt_hold_acct")
        if stock_ret is not None:
            t = stock_ret[stock_ret["account"] == acct].copy()
            t = t.sort_values("final_value", ascending=False)
            t = t.rename(columns={
                "ticker": "Ticker", "n_trades": "Trades",
                "invested": "Invested $", "final_value": "Final value $",
                "gain_pct": "Position gain %", "px_2000": "Px 2000",
                "px_2013": "Px 2013", "stock_return_pct": "Stock return %"})
            t = t.drop(columns=["account"])
            st.dataframe(
                t.style.format({"Invested $": "${:,.0f}",
                                "Final value $": "${:,.0f}",
                                "Position gain %": "{:+,.1f}%",
                                "Stock return %": "{:+,.1f}%",
                                "Px 2000": "{:.2f}", "Px 2013": "{:.2f}"},
                               na_rep="— (sold)")
                 .background_gradient(subset=["Final value $"], cmap="Greens"),
                use_container_width=True, height=600)
            top = t.iloc[0]
            st.caption(f"Biggest winner: **{top['Ticker']}** — "
                       f"${top['Invested $']:,.0f} invested became "
                       f"${top['Final value $']:,.0f}. Cash left uninvested: "
                       f"${trades[acct]['final_cash']:,.0f}.")
        _img(f"contrib_{acct}.png")
        _img(f"grid_{acct}.png",
             "Every holding normalized to 1.0 at Jan 2000 (grey = SPY)")

    # ---------------- 5. per-stock chart browser ----------------
    with sub[4]:
        stock_dir = os.path.join(CHARTS, "stocks")
        files = sorted(os.listdir(stock_dir)) if os.path.isdir(stock_dir) else []
        opts = [f[:-4] for f in files if f.endswith(".png")]
        if not opts:
            st.info("No per-stock charts found — run backtest/plot_suite.py.")
        else:
            labels = {o: f"{o.split('_')[0]}  ({NICE[o.split('_')[1]]})"
                      for o in opts}
            pick = st.selectbox("Pick a holding", opts,
                                format_func=lambda o: labels[o],
                                index=opts.index("ROST_growth")
                                if "ROST_growth" in opts else 0)
            _img(pick + ".png", subdir="stocks")
            st.caption("Monthly candles · orange = 40-week SMA · purple dashed "
                       "= DCF intrinsic value · ▲ green = buy/add at support "
                       "under IV · ▼ red = scandal sell · blue ▲ = replacement "
                       "buy · bottom panel = your position value.")
            tick, acct2 = pick.split("_")
            tt = [x for x in trades[acct2]["trades"] if x["ticker"] == tick]
            if tt:
                st.dataframe(pd.DataFrame(tt), use_container_width=True)

    # ---------------- 6. trade log ----------------
    with sub[5]:
        acct = st.radio("Account", ["growth", "defensive"], horizontal=True,
                        format_func=lambda k: NICE[k], key="bt_log_acct")
        tl = pd.DataFrame(trades[acct]["trades"])
        n_buy = (tl["action"] == "BUY").sum()
        n_add = (tl["action"] == "ADD").sum()
        n_sell = (tl["action"] == "SELL").sum()
        a, b, c, d = st.columns(4)
        a.metric("Initial buys", int(n_buy))
        b.metric("Support adds", int(n_add))
        c.metric("Scandal sells", int(n_sell))
        d.metric("Total capital deployed",
                 f"${tl[tl['action'] != 'SELL']['amount'].sum():,.0f}")
        st.dataframe(tl, use_container_width=True, height=520)
        st.caption("`pct_of_iv` = purchase price as % of that day's DCF "
                   "intrinsic value — every entry was made below IV.")

    # ---------------- 7. corrections ----------------
    with sub[6]:
        st.caption("Every peak-to-trough decline of **7%+**, with how long the "
                   "fall took and how long recovery took.")
        for k in ("growth", "defensive", "spy"):
            df = _corrections_df(stats, k)
            worst = df["Depth %"].min() if not df.empty else 0
            st.markdown(f"**{NICE[k]}** — {len(df)} corrections, "
                        f"worst {worst:.1f}%")
            if not df.empty:
                st.dataframe(
                    df.style.format({"Depth %": "{:+.1f}%"})
                      .background_gradient(subset=["Depth %"], cmap="Reds_r"),
                    use_container_width=True)
        _img("drawdown_compare.png",
             "Underwater chart — SPY spent most of the 14 years below a prior "
             "peak; the VMI books recovered far faster")

    # ---------------- 8. method ----------------
    with sub[7]:
        st.markdown("""
#### The rules (set in January 2000, never changed)
1. **Universe** — wide-moat great businesses, **zero dotcom/tech/telecom**.
   Defensive book: healthcare + staples franchises. Growth book: proven
   high-growth concepts (TJX, ROST, AZO, ORLY, SBUX, DHR, …).
2. **Valuation gate** — course-style 20-year DCF: base flow ≈ trailing EPS ×
   OCF-conversion; years 1–10 at analyst-era growth g, years 11–20 at 4%,
   **no terminal value**; discount rate = 6.5% (Jan-2000 10y Treasury) +
   β × 4% market-risk premium. **Buy only below intrinsic value.**
3. **Position sizing** — $1M per account, 16 names, 6.25% cap, bought in
   three tranches (~$20.8k each).
4. **Adds only at support** — tranches 2–3 require price ≤ 40-week SMA
   (weekly timeframe) **and** still under IV, minimum 8 weeks apart.
5. **Sell only on business deterioration/fraud** — BMY (2002 channel-stuffing)
   → PFE; UNH (2006 options backdating) → KO; CAH (2004 SEC probe) → GPC.
   MRK was **held** through Vioxx (product withdrawal ≠ fraud).
   Proceeds redeployed the same week.
6. IV refreshed continuously, growing at min(g, 10%)/yr.

#### Honest caveats
- Jan-2000 P/E, growth, beta and OCF-conversion inputs are **documented-era
  approximations**, not point-in-time database values.
- The defensive tilt is the one allowed piece of hindsight; ticker survival
  itself carries survivorship bias (FDO was unavailable → SYK).
- Adjusted prices mean dividends are reinvested per-stock.
- Scandal-sell dates use the actual news dates.
- UNH is the strategy's honest cost: sold on the backdating scandal, it went
  on to be a 9× stock — the rule still made sense *ex ante*.
""")
        st.caption("Artifacts: `backtest/simulate2.py` (engine) · "
                   "`trades.json` (structured log) · `stats2.json` · "
                   "`charts/` (48 charts) — all committed to the repo.")
