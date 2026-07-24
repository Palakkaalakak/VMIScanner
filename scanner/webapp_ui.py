"""VMI Great Business Scanner — Streamlit UI.

Run manually:  streamlit run scanner/webapp_ui.py
Or via PM2:    pm2 start ecosystem.scan.config.cjs --only vmi-streamlit
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    from scanner.i18n import tr
except ImportError:
    from i18n import tr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "public", "data", "scan_results.json")
ADHOC_PATH = "/tmp/adhoc_scan.json"

st.set_page_config(page_title="VMI Great Business Scanner", page_icon="📈",
                   layout="wide")

with st.sidebar:
    _it = st.toggle(tr("🇮🇹 Italiano"), value=(st.session_state.get("lang") == "it"),
                    key="lang_toggle",
                    help="OFF: English (default) · ON: Italiano — traduzione "
                         "con terminologia finanziaria italiana corretta")
    st.session_state["lang"] = "it" if _it else "en"

st.title(tr("📈 VMI Great Business Scanner"))
st.caption(tr("S&P 500 + Dow Jones 30 (toggle) · fundamentals-only checklist · SEC Company Facts primary, "
           "Yahoo + macrotrends fallbacks · trend/average checks require the "
           "full 20y window by default (toggle allows 20/15/10y any-pass) · "
           "IV = v13 sector-calibrated 20y DCF (StockOracle-matched: per-sector "
           "base-flow blend + fitted growth model, CAPM discount, + net cash; "
           "no terminal value · 36/36 calibration tickers within ±7%)"))


def run_scan(extra_args: list, label: str):
    """Run the scan CLI as a subprocess and stream stdout live."""
    cmd = [sys.executable, "-u", "-m", "scanner.vmi.scan"] + extra_args
    box = st.status(f"Running {label}…", expanded=True)
    log = box.empty()
    lines = []
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        lines.append(line.rstrip())
        log.code("\n".join(lines[-14:]), language=None)
    proc.wait()
    if proc.returncode == 0:
        box.update(label=f"{label} finished in {time.time()-t0:.0f}s ✅",
                   state="complete", expanded=False)
        # Commit + push immediately so a session wipe can't lose results.
        subprocess.run(["git", "add", "-A"], cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m",
                        f"Scan via UI {datetime.now(timezone.utc):%H:%M}Z"],
                       cwd=REPO_ROOT, capture_output=True)
        subprocess.run(["git", "push", "origin", "main", "-q"],
                       cwd=REPO_ROOT, capture_output=True)
    else:
        box.update(label=f"{label} FAILED (exit {proc.returncode})", state="error")
    st.rerun()


with st.sidebar:
    st.header(tr("Run scanner"))
    any_window = st.toggle(
        tr("Allow 20/15/10y any-pass"), value=False,
        help="OFF (default): trend/average checks must pass on the FULL "
             "20-year window. ON: the older lenient rule — pass if ANY of "
             "the 20y/15y/10y windows passes.")
    accept_5y = st.toggle(
        tr("Accept 5y-only passes"), value=False,
        help="ON: a trend/average check passes if the 5y window alone "
             "passes. OFF (default): 5y alone yields WARN — a long "
             "window must pass.")
    include_dow = st.toggle(
        tr("Include Dow Jones 30"), value=True,
        help="ON (default): scan universe = S&P 500 merged with the current "
             "Dow Jones Industrial Average 30 components (duplicates removed). "
             "OFF: S&P 500 only.")
    fresh = st.checkbox(tr("Force fresh data (ignore cache)"), value=False)
    rescore = st.checkbox(tr("Re-score all tickers (no resume)"), value=True)

    def _common_args():
        a = []
        if any_window:
            a.append("--any-long-window")
        if accept_5y:
            a.append("--accept-5y-alone")
        if not include_dow:
            a.append("--no-dow")
        return a

    scan_label = "S&P 500 + Dow 30" if include_dow else "S&P 500"
    if st.button(f"🚀 Run full {scan_label} scan", type="primary",
                 use_container_width=True):
        args = _common_args()
        if fresh:
            args.append("--no-cache")
        if rescore:
            args.append("--no-resume")
        run_scan(args, f"full {scan_label} scan (~1.5 min)")

    st.divider()
    st.subheader(tr("Scan specific tickers"))
    tickers_in = st.text_input(tr("Comma-separated tickers"), placeholder="AAPL, CNSWF, EVVTY")
    if st.button(tr("Scan tickers"), use_container_width=True) and tickers_in.strip():
        args = (["--tickers", tickers_in.replace(" ", ""), "--no-resume",
                 "--out", ADHOC_PATH] + _common_args())
        st.session_state["show_adhoc"] = True
        run_scan(args, f"scan of {tickers_in}")

def _render_ticker_detail(r):
    """Check-detail card for one scan-result dict (shared by the main
    Check-detail picker and the adhoc specific-ticker results)."""
    m = r.get("metrics") or {}
    st.markdown(f"**{r['ticker']} — {r.get('company','')}** · {r.get('sector','')} "
                f"/ {r.get('industry','')} · source: `{r.get('data_source','')}`")
    if r.get("error"):
        st.error(f"Scan error: {r['error']}")
        return
    if m.get("intrinsic_value") is not None:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(tr("Price"), f"${m.get('price'):,.2f}" if m.get("price") else "—")
        d2.metric(tr("Intrinsic value (DCF)"), f"${m['intrinsic_value']:,.2f}")
        disc = m.get("discount_pct")
        disc_txt = ("—" if disc is None
                    else f"({abs(disc):.1f}%)" if disc < 0 else f"{disc:.1f}%")
        d3.metric(tr("Discount"), disc_txt,
                  help="x% = trading below intrinsic value; (x%) = premium above IV")
        d4.metric(tr("DCF growth used"), f"{m.get('dcf_growth_used', 0):.1f}%/yr")
    checks = pd.DataFrame([{
        "Check": ch["name"],
        "Status": {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                   "WARN": "⚠️ WARN", "NA": "➖ NA"}.get(ch["status"], ch["status"]),
        "Value": ch.get("value", ""), "Note": ch.get("detail", ""),
    } for ch in r.get("checks", [])])
    st.dataframe(checks, use_container_width=True, height=530)


def _verdict(r):
    if r.get("is_great"):
        return "✅ GREAT"
    return "🟡 NEAR" if r.get("n_fail", 9) <= 1 else "❌ FAIL"


# ---- Top-level tabs: Scanner (existing) + Backtest dashboard ---------
# Big, obvious tab buttons (default streamlit tabs are easy to miss).
st.markdown("""
<style>
div[data-testid="stTabs"] > div > div[role="tablist"] button[role="tab"] {
    font-size: 1.25rem; font-weight: 700; padding: 0.9rem 1.6rem;
    background: #f0f2f6; border-radius: 10px 10px 0 0; margin-right: 6px;
}
div[data-testid="stTabs"] > div > div[role="tablist"] button[aria-selected="true"] {
    background: #e8f0fe; border-bottom: 4px solid #1a73e8;
}
</style>""", unsafe_allow_html=True)
tab_scan, tab_bt = st.tabs([tr("🔎 Scanner"), tr("🕰️ Backtest 2000–2013")])

with tab_bt:
    try:
        from scanner import backtest_tab
    except ImportError:  # streamlit puts scanner/ itself on sys.path
        import backtest_tab
    backtest_tab.render()

with tab_scan:
    # ---- Adhoc "Scan specific tickers" results (survives st.rerun via
    # ---- session_state; reads the /tmp output the adhoc scan wrote) --------
    if st.session_state.get("show_adhoc") and os.path.exists(ADHOC_PATH):
        try:
            with open(ADHOC_PATH) as f:
                adhoc = json.load(f)
            adhoc_rows = adhoc.get("results", [])
        except (json.JSONDecodeError, OSError):
            adhoc_rows = []
        if adhoc_rows:
            hdr, btn = st.columns([5, 1])
            hdr.subheader("🎯 Specific-ticker scan results")
            if btn.button(tr("Dismiss"), key="dismiss_adhoc"):
                st.session_state["show_adhoc"] = False
                st.rerun()
            ts = adhoc.get("generated_at", "")[:19].replace("T", " ")
            st.caption(f"Scanned {ts} UTC · these results are shown here only — "
                       "they are NOT merged into the main scan table below.")
            ok_rows = [r for r in adhoc_rows if not r.get("error")]
            if ok_rows:
                summary = pd.DataFrame([{
                    "Ticker": r["ticker"], "Company": r.get("company", ""),
                    "Verdict": _verdict(r), "Fails": r.get("n_fail", 0),
                    "Warns": r.get("n_warn", 0), "Score": r.get("score", 0),
                    "Price $": (r.get("metrics") or {}).get("price"),
                    "Intrinsic Value $": (r.get("metrics") or {}).get("intrinsic_value"),
                    "Discount %": (r.get("metrics") or {}).get("discount_pct"),
                    "Source": r.get("data_source", ""),
                } for r in ok_rows])
                st.dataframe(summary, use_container_width=True, hide_index=True,
                             column_config={
                                 "Price $": st.column_config.NumberColumn(format="dollar"),
                                 "Intrinsic Value $": st.column_config.NumberColumn(format="dollar"),
                                 "Discount %": st.column_config.NumberColumn(format="%.1f%%")})
            for r in sorted(adhoc_rows, key=lambda x: x["ticker"]):
                with st.expander(f"{r['ticker']} — "
                                 f"{_verdict(r) if not r.get('error') else '⚠️ ERROR'}",
                                 expanded=(len(adhoc_rows) == 1)):
                    _render_ticker_detail(r)
            st.divider()

    if not os.path.exists(RESULTS_PATH):
        st.info(tr("No results yet — hit **Run full scan** in the sidebar "
                "(S&P 500 + Dow 30 by default)."))
        st.stop()

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    c = data.get("counts", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(tr("✅ Great"), c.get("great", 0))
    m2.metric(tr("🟡 Near miss (1 fail)"), c.get("near_miss", 0))
    m3.metric(tr("❌ Failed"), c.get("failed", 0))
    m4.metric(tr("Errors"), c.get("errors", 0))
    m5.metric(tr("Excluded"), data.get("excluded_count", 0),
              help="ETFs / banks / REITs — VMI exception rules need data we don't have")
    st.caption(f"Last scan: {data.get('generated_at', '')[:19].replace('T', ' ')} UTC "
               f"· universe {data.get('universe_size', '?')} tickers "
               f"({data.get('universe', 'S&P 500')})")

    rows = [r for r in data["results"] if not r.get("error")]


    # ---- Build the master dataframe: display columns + ALL known data as
    # ---- extra (hideable) columns so the custom filter can use everything.

    # Fraction-stored metrics get converted to % for display/filtering.
    _FRACTION_METRICS = {"rev_cagr_5y", "rev_cagr_10y", "rev_cagr_15y",
                         "ni_cagr_5y", "ni_cagr_10y", "ni_cagr_15y",
                         "cfo_cagr_10y"}
    _METRIC_LABELS = {
        "price": "Price $",
        "intrinsic_value": "Intrinsic Value $",
        "discount_pct": "Discount %",
        "dcf_growth_used": "DCF growth used %",
        "rev_cagr_5y": "Rev CAGR 5y %", "rev_cagr_10y": "Rev CAGR 10y %",
        "rev_cagr_15y": "Rev CAGR 15y %",
        "ni_cagr_5y": "NI CAGR 5y %", "ni_cagr_10y": "NI CAGR 10y %",
        "ni_cagr_15y": "NI CAGR 15y %",
        "cfo_cagr_10y": "CFO CAGR 10y %",
        "proj_eps_next_5y": "Proj EPS 5y %/yr",
        "proj_eps_next_y": "Proj EPS next-Y %",
        "eps_past_5y": "EPS past 5y %/yr",
    }

    all_metric_keys = sorted({k for r in rows for k in (r.get("metrics") or {})})


    def _row(r):
        m = r.get("metrics") or {}
        d = {
            "Ticker": r["ticker"], "Company": r.get("company", ""),
            "Sector": r.get("sector", ""), "Verdict": _verdict(r),
            "Fails": r.get("n_fail", 0), "Warns": r.get("n_warn", 0),
            "Score": r.get("score", 0),
            "Applicable checks": r.get("applicable", None),
            "Source": r.get("data_source", ""),
        }
        for k in all_metric_keys:
            v = m.get(k)
            if v is not None and k in _FRACTION_METRICS:
                v = round(v * 100, 1)
            d[_METRIC_LABELS.get(k, k)] = v
        return d


    df = pd.DataFrame([_row(r) for r in rows])


    # Main visible columns. "Discount %" stays NUMERIC so that clicking the
    # column header sorts by value (a formatted string would sort
    # alphabetically — that was the old "seemingly random" ordering).
    MAIN_COLS = ["Ticker", "Company", "Sector", "Verdict", "Fails", "Warns",
                 "Score", "Price $", "Intrinsic Value $", "Discount %",
                 "Source"]
    MAIN_COLS = [col for col in MAIN_COLS if col in df.columns]

    _COL_CONFIG = {
        "Price $": st.column_config.NumberColumn(format="dollar"),
        "Intrinsic Value $": st.column_config.NumberColumn(format="dollar"),
        "Discount %": st.column_config.NumberColumn(
            format="%.1f%%",
            help="Positive = trading below intrinsic value; "
                 "negative = premium above IV"),
    }

    # ---- Base filters -----------------------------------------------------
    fc1, fc2, fc3 = st.columns([2, 2, 3])
    verdict_f = fc1.multiselect(tr("Verdict"), ["✅ GREAT", "🟡 NEAR", "❌ FAIL"],
                                default=["✅ GREAT"])
    sector_f = fc2.multiselect(tr("Sector"), sorted(x for x in df["Sector"].unique() if x))
    search = fc3.text_input(tr("Search ticker / company"))

    view = df[df["Verdict"].isin(verdict_f)] if verdict_f else df
    if sector_f:
        view = view[view["Sector"].isin(sector_f)]
    if search:
        s = search.strip().lower()
        view = view[view["Ticker"].str.lower().str.contains(s)
                    | view["Company"].str.lower().str.contains(s)]

    # ---- Custom filters + sorting (ALL known data) ------------------------
    numeric_fields = sorted(
        col for col in df.columns
        if col not in ("Ticker", "Company", "Sector", "Verdict", "Source")
        and pd.api.types.is_numeric_dtype(df[col]))

    with st.expander(tr("🔧 Custom filters & sorting (all known data)"), expanded=False):
        st.caption(tr("Stack any number of numeric filters on top of the verdict "
                   "filter above. Blank rows in the data (NA) are **kept** by "
                   "default — NA never disqualifies — untick to drop them."))
        n_filters = st.number_input(tr("Number of custom filters"), 0, 8, 0)
        for i in range(int(n_filters)):
            f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
            field = f1.selectbox(f"Field #{i+1}", numeric_fields, key=f"ff{i}")
            col_data = df[field].dropna()
            lo_default = float(col_data.min()) if len(col_data) else 0.0
            hi_default = float(col_data.max()) if len(col_data) else 0.0
            lo = f2.number_input("Min", value=lo_default, key=f"lo{i}")
            hi = f3.number_input("Max", value=hi_default, key=f"hi{i}")
            keep_na = f4.checkbox(tr("Keep NA"), value=True, key=f"na{i}")
            mask = (view[field] >= lo) & (view[field] <= hi)
            if keep_na:
                mask = mask | view[field].isna()
            view = view[mask]

        st.divider()
        s1, s2 = st.columns([3, 2])
        sort_by = s1.selectbox(tr("Sort by"), ["(none)"] + numeric_fields
                               + ["Ticker", "Company", "Sector"])
        sort_dir = s2.radio(tr("Direction"),
                            [tr("Descending"), tr("Ascending")],
                            horizontal=True)
        if sort_by != "(none)":
            view = view.sort_values(
                sort_by, ascending=(sort_dir == tr("Ascending")),
                na_position="last", kind="mergesort")

        show_all_cols = st.checkbox(
            tr("Show ALL data columns in the table (CAGRs, projections, …)"),
            value=False)

    st.caption(tr("{n} stocks shown · click any column header to sort "
                  "ascending/descending · Discount % > 0 = below intrinsic "
                  "value, < 0 = premium").replace("{n}", str(len(view))))
    table = view if show_all_cols else view[MAIN_COLS]
    st.dataframe(table.reset_index(drop=True), use_container_width=True,
                 height=460, column_config=_COL_CONFIG, hide_index=True)

    st.subheader(tr("Check detail"))
    pick = st.selectbox(tr("Ticker"), [""] + view["Ticker"].tolist())
    if pick:
        r = next(x for x in rows if x["ticker"] == pick)
        _render_ticker_detail(r)
        st.caption(tr("NA = data not reported by the source or check not applicable "
                   "to this company type — NA never disqualifies a stock."))
