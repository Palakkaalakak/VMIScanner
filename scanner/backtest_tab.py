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
        "growth_cc": "🚀 Growth + covered calls",
        "defensive_pmcc": "🛡️ Defensive PMCC (full pyramid)",
        "growth_pmcc": "🚀 Growth PMCC (full pyramid)",
        "defensive_pmcc_hp": "🛡️ Defensive PMCC (half-pyramid)",
        "growth_pmcc_hp": "🚀 Growth PMCC (half-pyramid)",
        "defensive_pmcc_conv": "🛡️ Defensive PMCC→shares",
        "growth_pmcc_conv": "🚀 Growth PMCC→shares"}


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
    for k in ("defensive", "growth"):
        for v in ("pmcc", "pmcc_hp", "pmcc_conv"):
            p = os.path.join(BT, f"eq_{k}_{v}.csv")
            if os.path.exists(p):
                eq[f"{k}_{v}"] = pd.read_csv(p, index_col=0,
                                             parse_dates=True).iloc[:, 0]
    stats = json.load(open(os.path.join(BT, "stats2.json")))
    trades = json.load(open(os.path.join(BT, "trades.json")))

    def _j(name):
        p = os.path.join(BT, name)
        return json.load(open(p)) if os.path.exists(p) else None

    cc_stats = _j("stats_cc_2000_2013.json")
    opt_stats = _j("stats_options_2000_2013.json")
    sweep = _j("cc_threshold_sweep.json")
    sweep_eras = _j("cc_sweep_eras.json")
    ext26 = _j("stats_options_2000_2026.json")
    ret_path = os.path.join(CHARTS, "stock_returns.csv")
    stock_ret = pd.read_csv(ret_path) if os.path.exists(ret_path) else None
    return (eq, stats, trades, stock_ret, cc_stats, opt_stats,
            sweep, sweep_eras, ext26)


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

    (eq, stats, trades, stock_ret, cc_stats, opt_stats,
     sweep, sweep_eras, ext26) = _load()

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

    if cc_stats:
        with st.container(border=True):
            st.markdown("**➕ Same books, but selling covered calls on the "
                        "CC-viable names** — monthly ~Δ0.42 calls sold only "
                        "on stocks with growth ≤ 15%; fast growers held "
                        "untouched (no calls).")
            c1, c2 = st.columns(2)
            for col, k in ((c1, "growth"), (c2, "defensive")):
                s = cc_stats[k]
                col.metric(NICE[k + "_cc"],
                           f"${s['final']:,.0f}",
                           f"{s['cagr_pct']}% CAGR · max DD "
                           f"{s['max_dd_pct']}% · calls on "
                           f"{len(s['cc_names'])}/16 names")

    sub = st.tabs(["📈 Equity curves", "🕯️ Candles", "📊 Year by year",
                   "🧱 Holdings & contribution", "🔍 Stock charts",
                   "📜 Trade log", "🌊 Corrections",
                   "🎯 Options overlays", "🛒 Today's portfolio (2026)",
                   "🧪 Method & caveats"])

    # ---------------- 1. equity curves (interactive) ----------------
    with sub[0]:
        c1, c2 = st.columns([1, 1])
        log_scale = c1.toggle("Log scale", value=False,
                              help="Log scale shows compounding consistency — "
                                   "straight line = constant CAGR.")
        norm = c2.toggle("Normalize to 1.0", value=False)
        keys = ["defensive", "growth", "spy"]
        colors = ["#1f77b4", "#2ca02c", "#888888"]
        has_cc = "defensive_cc" in eq and "growth_cc" in eq
        show_cc = False
        if has_cc:
            show_cc = st.toggle(
                "Overlay: sell covered calls on the CC-viable names",
                value=True,
                help="Same books, same value-gated entries — but monthly "
                     "~delta-0.42 calls are sold on every stock whose growth "
                     "is ≤ 15% (all 16 defensive names; 9 of 16 growth "
                     "names). Fast growers are held as plain shares with no "
                     "calls.")
            if show_cc:
                keys = ["defensive", "defensive_cc", "growth", "growth_cc",
                        "spy"]
                colors = ["#1f77b4", "#7fbfef", "#2ca02c", "#98df8a",
                          "#888888"]
        has_pmcc = "defensive_pmcc" in eq
        if has_pmcc:
            show_pmcc = st.toggle(
                "Overlay: PMCC (deep-ITM long calls instead of shares, "
                "leveraged)", value=False,
                help="Same value/tranche rules, but CC-viable names are "
                     "held via delta-80 LEAPS calls with monthly short "
                     "calls on top (natural ~4-5\u00d7 leverage). Spend per name "
                     "capped at 6.25% of capital \u2014 in dollars spent, not "
                     "exposure. WARNING: huge drawdowns.")
            if show_pmcc:
                keys = keys[:-1] + ["defensive_pmcc_conv", "growth_pmcc",
                                    "spy"]
                colors = colors[:-1] + ["#d62728", "#ff7f0e", "#888888"]
        df = pd.DataFrame({NICE[k]: eq[k] for k in keys})
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
                                                range=colors)),
                tooltip=["Date:T", "Portfolio:N",
                         alt.Tooltip("Value:Q", format=",.0f")])
                .properties(height=430).interactive())
            st.altair_chart(ch, use_container_width=True)
        except Exception:
            st.line_chart(df, height=430)
        st.caption("Hover for values · drag to zoom · double-click to reset.")
        if has_cc and show_cc and cc_stats:
            st.caption(
                "**Covered-call variant note:** the CC curves are a "
                "day-by-day simulation on unadjusted prices with dividends "
                "credited as cash and Black-Scholes call pricing at each "
                "stock's realized volatility, while the base curves use "
                "adjusted prices (dividends auto-reinvested). Same "
                "economics, slightly different bookkeeping — so compare "
                "each CC curve to the index and to its own start, not "
                "penny-for-penny against its base twin. Growth book calls "
                "were sold on: "
                + ", ".join(cc_stats["growth"]["cc_names"])
                + ". The other 7 fast growers were held call-free.")
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
        yr_keys = ["defensive", "growth", "spy"]
        if "defensive_cc" in eq and st.toggle(
                "Include covered-call variants", value=False,
                key="bt_yr_cc"):
            yr_keys = ["defensive", "defensive_cc", "growth", "growth_cc",
                       "spy"]
        yr = pd.DataFrame({k: eq[k].resample("YE").last()
                           for k in yr_keys})
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

    # ---------------- 8. options overlays ----------------
    with sub[7]:
        if not (cc_stats or opt_stats):
            st.info("Run `python backtest/cc_2000_2013.py` and "
                    "`python backtest/options_2000_2013.py` to generate the "
                    "options-overlay artifacts.")
        if cc_stats:
            st.markdown("#### ① Covered calls on the CC-viable names")
            st.markdown(
                "**The rule (repeatable, no hindsight):** sell monthly "
                "~Δ0.42 calls only on names whose *book growth rate* "
                "g ≤ 15%; hold faster growers untouched. The growth input "
                "is the same era-documented figure already used by the DCF "
                "— no new data needed.")
            rows = []
            for k in ("defensive", "growth"):
                base = stats[k]
                s = cc_stats[k]
                rows.append({
                    "Book": NICE[k],
                    "Plain CAGR": f"{base['cagr_pct']}%",
                    "With CC CAGR": f"{s['cagr_pct']}%",
                    "Plain final": f"${base['final_value']:,.0f}",
                    "With CC final": f"${s['final']:,.0f}",
                    "CC max DD": f"{s['max_dd_pct']}%",
                    "Calls on": f"{len(s['cc_names'])}/16"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
            st.caption(
                "Growth-book names that got calls: "
                + ", ".join(cc_stats["growth"]["cc_names"])
                + " · held call-free: "
                + ", ".join(cc_stats["growth"]["plain_names"]) + ".")
        if sweep:
            st.markdown("#### ② Is the 15% cutoff cherry-picked? "
                        "(threshold sweep)")
            st.markdown(
                "Same 2000–2013 sim with the cutoff moved. **Honest "
                "verdict: in THIS window, selling calls on everything was "
                "better** — 2000–2013 was a sideways market, covered "
                "calls’ best weather. The cutoff is there for trending "
                "markets, where capping a compounder costs more than the "
                "premium collected.")
            sw_rows = []
            for thr, d in sweep["growth"].items():
                lab = "all 16" if float(thr) > 1 else f"g ≤ {float(thr):.0%}"
                sw_rows.append({"Cutoff": lab, "Names with calls": d["n_cc"],
                                "CAGR": f"{d['cagr_pct']}%",
                                "Max DD": f"{d['max_dd_pct']}%"})
            st.dataframe(pd.DataFrame(sw_rows), use_container_width=True,
                         hide_index=True)
            st.caption("Growth book, 2000–2013. The defensive book is "
                       "insensitive — all 16 names are already ≤ 15%.")
        if sweep_eras:
            st.markdown("**Cross-era check (Dow-style books held to "
                        "2026):** CAGR by cutoff")
            er_rows = []
            for yr_, d in sweep_eras.items():
                r = {"Start": yr_}
                for thr, v in d.items():
                    lab = ("all" if float(thr) > 1
                           else f"g≤{float(thr):.0%}")
                    r[lab] = f"{v['cagr_pct']}%"
                er_rows.append(r)
            st.dataframe(pd.DataFrame(er_rows), use_container_width=True,
                         hide_index=True)
            st.caption(
                "The pattern repeats in every era on these value books — "
                "more calls, more return — because IV-gated entries buy "
                "cheap, rarely-runaway names. The cutoff matters most when "
                "the book holds true hyper-growers (SBUX/DLTR/ORLY at "
                "22–27% growth).")
        if opt_stats:
            st.markdown("#### ③ PMCC instead of shares (leveraged)")
            st.markdown(
                "Same allocation rules — **6.25% per name in dollars "
                "SPENT** (not exposure), 3 tranches, 56-day gaps, IV-gated "
                "— but CC-viable names are held as deep-ITM (Δ0.80) LEAPS "
                "calls with monthly short calls on top. A delta-80 call "
                "costs ~20–25% of the stock, so each dollar controls "
                "~4–5× the shares. **The leverage remains**: you may "
                "spend at most 6.25% of the CURRENT account value buying "
                "any one name's calls (on $1M that's $62.5k controlling "
                "~$250–300k of stock). The cap grows with the account, so "
                "**all premium income is reinvested** — it never "
                "dead-piles as cash — and a position that outgrows 6.25% "
                "on its own is NEVER trimmed.")
            prows = []
            for k in ("defensive", "growth"):
                for v, nm in (("pmcc", "full pyramid"),
                              ("pmcc_hp", "half-pyramid"),
                              ("pmcc_conv", "convert→shares")):
                    d = opt_stats.get(k, {}).get(v)
                    if d:
                        prows.append({
                            "Book": NICE[k], "Variant": nm,
                            "CAGR": f"{d['cagr_pct']}%",
                            "Max DD": f"{d['max_dd_pct']}%",
                            "Final": f"${d['final']:,.0f}"})
            if prows:
                st.dataframe(pd.DataFrame(prows), use_container_width=True,
                             hide_index=True)
            st.warning(
                "Leverage cuts both ways: drawdowns run −30% to −90%, and "
                "Black-Scholes pricing understates real-world spreads, "
                "assignment risk and margin calls. The convert→shares "
                "variant (bank the leveraged gain, become a plain "
                "shareholder when the long call goes deep ITM) is the only "
                "PMCC flavor with a survivable risk profile.")
        if ext26:
            st.markdown("#### ④ Structural test — same rules held to 2026")
            st.markdown(
                "If the overlays were only good for the sideways "
                "2000–2013 decade, the 2013–2026 bull leg would expose "
                "them. Same books, same rules, run straight through:")
            xrows = []
            for k in ("defensive", "growth"):
                for v, nm in (("plain", "buy & hold"),
                              ("cc", "covered calls"),
                              ("pmcc_conv", "PMCC→shares"),
                              ("pmcc_hp", "PMCC half-pyramid"),
                              ("pmcc", "PMCC full pyramid")):
                    d = ext26.get(k, {}).get(v)
                    if d:
                        xrows.append({
                            "Book": NICE[k], "Variant": nm,
                            "CAGR 2000–2026": f"{d['cagr_pct']}%",
                            "CAGR 00–13": f"{d.get('cagr_2000_2013','—')}%",
                            "CAGR 13–26": f"{d.get('cagr_2013_2026','—')}%",
                            "Max DD": f"{d['max_dd_pct']}%",
                            "Final ($1M start)": f"${d['final']:,.0f}"})
            st.dataframe(pd.DataFrame(xrows), use_container_width=True,
                         hide_index=True)
            st.caption(
                "**Verdict: structural.** Every overlay beats its own "
                "buy-and-hold book in BOTH sub-periods, not just the lost "
                "decade. The edge shrinks in the trending 2013–26 leg "
                "(covered calls: ~+12pp → ~+5pp over plain) exactly as "
                "theory predicts — call-selling pays best sideways — but "
                "it never flips negative. Sub-period CAGRs use the same "
                "single continuous run split at 2013-12-27.")

    # ---------------- 9. method ----------------
    with sub[8]:
        _pf_path = os.path.join(BT, "portfolio_2026.json")
        pf = (json.load(open(_pf_path))
              if os.path.exists(_pf_path) else None)
        if not pf:
            st.info("Run the portfolio selection to generate "
                    "`backtest/portfolio_2026.json`.")
        else:
            st.markdown("#### The 2026 Anti-Bubble Bedrock book — 16 "
                        "wide-moat greats, picked from the live scanner "
                        f"({pf['source'].split('(')[1].rstrip(')')})")
            st.caption("Workflow: scanner DCF first → manual moat "
                       "verification second. Bottom-heavy on the "
                       "bedrock pyramid — Healthcare (5) › Financial "
                       "toll booths (3) › SaaS/vertical software (3) › "
                       "Comms (2) › Consumer compounders (2) › "
                       "Industrial services (1) › Materials (0 — LIN/"
                       "ECL/SHW at fair value, Tier-3 'wait for the "
                       "discount'). No commodity knife-fighters "
                       "(INTC/AMD class rejected — no sustainable "
                       "advantage). g ≤ 15% gets the PMCC overlay; "
                       "faster growers stay plain shares.")
            rows = pf["portfolio"]
            pdf = pd.DataFrame([{
                "Ticker": r["ticker"],
                "Bedrock tier": r.get("tier", ""),
                "Strategy": ("🎯 PMCC" if r["type"] == "PMCC"
                             else "📈 Plain shares"),
                "Proj. EPS growth %": r["g5"],
                "Price $": r["price"],
                "Intrinsic value $": r["iv"],
                "Discount %": r["discount_pct"],
                "Moat (manually verified)": r["moat"],
            } for r in rows])
            st.dataframe(pdf, use_container_width=True, hide_index=True,
                         height=620)
            c = pf["counts"]
            st.success(f"**Action today:** all 16 names trade below "
                       f"intrinsic value → open **tranche 1 in every "
                       f"name now**. {c['pmcc']} PMCC names (buy "
                       f"deep-ITM Δ0.80 LEAPS, spend ≤ 6.25% of account "
                       f"value on premium, sell Δ0.42 monthlies, roll "
                       f"shorts at Δ0.80) · {c['plain']} plain-share "
                       f"names (⅓ of the 6.25% slot now). Adds: 200-day "
                       f"SMA ×1.01 while under IV, ≥ 56 days apart. "
                       f"Never trim winners.")
            st.warning("**Data honesty:** scanner names with discounts "
                       "above ~80% (WDAY, DPZ, YUM, MNST, HSY, DG, RL, "
                       "MU, DECK, GRMN, CTSH, IT) look like projection/"
                       "split artifacts and were **excluded** rather "
                       "than trusted; ZTS (70.7%) also benched as "
                       "borderline. The universe now includes the "
                       "Nasdaq-100 + curated extras (540 tickers): "
                       "POOL scanned GREAT (34.6% discount) and joined "
                       "the book; MELI/CNSWF are near-misses (WARN "
                       "flags), NVO fails checks outright — no DCF "
                       "pass, no buy (rules first). ADBE and INTU "
                       "are cheap partly *because* the market fears AI "
                       "disruption — that is the value bet, stated "
                       "openly. MSFT/META are the only AI-adjacent "
                       "names, kept because they sit in the bedrock "
                       "with diversified cash engines. All prices/IVs "
                       "from the 2026-07-22 scan; re-check before "
                       "placing orders.")

    # ---------------- 10. method & caveats ----------------
    with sub[9]:
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

#### Options overlays — method
- **Covered calls**: monthly ~Δ0.42 calls, rolled when Δ reaches 0.80,
  written only on names with book growth g ≤ 15% (the CC-viable rule).
  Premiums are reinvested into the cheapest below-IV book stock.
- **PMCC**: Δ0.80 LEAPS (~150 DTE) instead of shares on CC-viable names,
  Δ0.42 short calls (~35 DTE) on top; long rolled at 30 DTE. New capital
  per name is capped at 6.25% of the *current account value*, in premium
  dollars spent — the cap limits what you PAY, not what you control
  ($62.5k of premium buys calls on ~$250–300k of stock; the leverage
  remains). Because the cap grows with the account, every dollar of
  premium income is reinvestable — premiums land in cash, count in the
  account value, and are redeployed into the cheapest below-IV name.
  Once a position outgrows 6.25% by itself it is never trimmed — exactly
  mirroring how a stock position is never cut for outgrowing its weight.
  Rolls recycle a lot's own sale proceeds and do not count as new
  capital.
- **Pricing caveat**: real options history does not exist back to 2000, so
  all option prices are **Black-Scholes at each stock's realized
  volatility**. This ignores bid/ask spreads, early assignment, the vol
  smile and margin — treat PMCC numbers as upper bounds, not
  expectations.
- **Bookkeeping caveat**: overlay sims run on unadjusted prices with cash
  dividends credited explicitly; the base curves use adjusted prices
  (dividends auto-reinvested). Same economics — compare shapes and
  end-points, not pennies.
- **PMCC forfeits dividends** on the option-held names — that, not time
  decay, is its structural cost on these dividend-heavy books.
""")
        st.caption("Artifacts: `backtest/simulate2.py` (engine) · "
                   "`trades.json` (structured log) · `stats2.json` · "
                   "`charts/` (48 charts) — all committed to the repo.")
