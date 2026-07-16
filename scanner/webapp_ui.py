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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "public", "data", "scan_results.json")

st.set_page_config(page_title="VMI Great Business Scanner", page_icon="📈",
                   layout="wide")
st.title("📈 VMI Great Business Scanner")
st.caption("S&P 500 · fundamentals-only (no valuation/TA) · SEC Company Facts "
           "primary, Yahoo + macrotrends fallbacks · trend/average checks pass "
           "on ANY of 20/15/10y windows")


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
    st.header("Run scanner")
    accept_5y = st.toggle(
        "Accept 5y-only passes", value=False,
        help="ON: a trend/average check passes if ANY window (20/15/10y or "
             "just 5y) passes. OFF (default): 5y alone yields WARN — a long "
             "window (20/15/10y) must pass.")
    fresh = st.checkbox("Force fresh data (ignore cache)", value=False)
    rescore = st.checkbox("Re-score all tickers (no resume)", value=True)
    if st.button("🚀 Run full S&P 500 scan", type="primary", use_container_width=True):
        args = []
        if accept_5y:
            args.append("--accept-5y-alone")
        if fresh:
            args.append("--no-cache")
        if rescore:
            args.append("--no-resume")
        run_scan(args, "full S&P 500 scan (~1.5 min)")

    st.divider()
    st.subheader("Scan specific tickers")
    tickers_in = st.text_input("Comma-separated tickers", placeholder="AAPL, CNSWF, EVVTY")
    if st.button("Scan tickers", use_container_width=True) and tickers_in.strip():
        args = ["--tickers", tickers_in.replace(" ", ""), "--no-resume",
                "--out", "/tmp/adhoc_scan.json"]
        if accept_5y:
            args.append("--accept-5y-alone")
        run_scan(args, f"scan of {tickers_in}")

if not os.path.exists(RESULTS_PATH):
    st.info("No results yet — hit **Run full S&P 500 scan** in the sidebar.")
    st.stop()

with open(RESULTS_PATH) as f:
    data = json.load(f)

c = data.get("counts", {})
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("✅ Great", c.get("great", 0))
m2.metric("🟡 Near miss (1 fail)", c.get("near_miss", 0))
m3.metric("❌ Failed", c.get("failed", 0))
m4.metric("Errors", c.get("errors", 0))
m5.metric("Excluded", data.get("excluded_count", 0),
          help="ETFs / banks / REITs — VMI exception rules need data we don't have")
st.caption(f"Last scan: {data.get('generated_at', '')[:19].replace('T', ' ')} UTC "
           f"· universe {data.get('universe_size', '?')} tickers")

rows = [r for r in data["results"] if not r.get("error")]


def _verdict(r):
    if r.get("is_great"):
        return "✅ GREAT"
    return "🟡 NEAR" if r.get("n_fail", 9) <= 1 else "❌ FAIL"


df = pd.DataFrame([{
    "Ticker": r["ticker"], "Company": r.get("company", ""),
    "Sector": r.get("sector", ""), "Verdict": _verdict(r),
    "Fails": r.get("n_fail", 0), "Warns": r.get("n_warn", 0),
    "Score": r.get("score", 0),
    "Rev CAGR 5y %": (r.get("metrics") or {}).get("rev_cagr_5y"),
    "Rev CAGR 10y %": (r.get("metrics") or {}).get("rev_cagr_10y"),
    "Rev CAGR 15y %": (r.get("metrics") or {}).get("rev_cagr_15y"),
    "NI CAGR 5y %": (r.get("metrics") or {}).get("ni_cagr_5y"),
    "NI CAGR 10y %": (r.get("metrics") or {}).get("ni_cagr_10y"),
    "NI CAGR 15y %": (r.get("metrics") or {}).get("ni_cagr_15y"),
    "Proj EPS 5y %/yr": (r.get("metrics") or {}).get("proj_eps_next_5y"),
    "Proj EPS next-Y %": (r.get("metrics") or {}).get("proj_eps_next_y"),
    "Source": r.get("data_source", ""),
} for r in rows])

# CAGRs are stored as fractions (0.112 = 11.2%); projections already percent.
for col in df.columns:
    if "CAGR" in col:
        df[col] = (df[col] * 100).round(1)
    elif col.startswith("Proj"):
        df[col] = df[col].round(1)

fc1, fc2, fc3 = st.columns([2, 2, 3])
verdict_f = fc1.multiselect("Verdict", ["✅ GREAT", "🟡 NEAR", "❌ FAIL"],
                            default=["✅ GREAT"])
sector_f = fc2.multiselect("Sector", sorted(x for x in df["Sector"].unique() if x))
search = fc3.text_input("Search ticker / company")

view = df[df["Verdict"].isin(verdict_f)] if verdict_f else df
if sector_f:
    view = view[view["Sector"].isin(sector_f)]
if search:
    s = search.strip().lower()
    view = view[view["Ticker"].str.lower().str.contains(s)
                | view["Company"].str.lower().str.contains(s)]

st.caption("Click any column header to sort ascending/descending "
           "(CAGRs, projected growth, score…)")
st.dataframe(view.reset_index(drop=True), use_container_width=True, height=460)

st.subheader("Check detail")
pick = st.selectbox("Ticker", [""] + view["Ticker"].tolist())
if pick:
    r = next(x for x in rows if x["ticker"] == pick)
    st.markdown(f"**{r['ticker']} — {r.get('company','')}** · {r.get('sector','')} "
                f"/ {r.get('industry','')} · source: `{r.get('data_source','')}`")
    checks = pd.DataFrame([{
        "Check": ch["name"],
        "Status": {"PASS": "✅ PASS", "FAIL": "❌ FAIL",
                   "WARN": "⚠️ WARN", "NA": "➖ NA"}.get(ch["status"], ch["status"]),
        "Value": ch.get("value", ""), "Note": ch.get("detail", ""),
    } for ch in r.get("checks", [])])
    st.dataframe(checks, use_container_width=True, height=530)
    st.caption("NA = data not reported by the source or check not applicable "
               "to this company type — NA never disqualifies a stock.")
