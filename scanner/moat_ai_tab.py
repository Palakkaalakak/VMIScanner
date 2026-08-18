"""AI Moat Evaluator tab — runs YOUR fine-tuned moat model via LM Studio.

How it works
------------
* LM Studio (on the machine running this dashboard) serves the trained
  moat model over its OpenAI-compatible local server (default
  http://localhost:1234/v1).
* This tab detects the server, auto-picks the model whose name contains
  "moat" (the quantize step names the file moat-<base>-<quant>.gguf),
  and sends it the EXACT same prompt format used in training:
  rubric system prompt + the scanner's quantitative evidence card.
* Every evaluation is saved to public/data/moat_ai_evaluations.json —
  so verdicts/scores survive restarts, appear as filterable columns in
  the Scanner tab, and (once committed) travel with the repo.

IMPORTANT: the dashboard's Python talks to LM Studio, not your browser.
If the dashboard runs in the cloud sandbox, it CANNOT reach the LM
Studio on your PC — run the dashboard locally for live evaluation:
    streamlit run scanner/webapp_ui.py
Saved evaluations are visible everywhere either way.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    from scanner.i18n import tr
except ImportError:
    from i18n import tr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "public", "data", "scan_results.json")
EVALS_PATH = os.path.join(REPO_ROOT, "public", "data",
                          "moat_ai_evaluations.json")
RUBRIC_PATH = os.path.join(REPO_ROOT, "ai_moat", "rubric_system_prompt.md")

DEFAULT_BASE_URL = "http://localhost:1234/v1"

# Same first-line format the model was trained to emit:
#   "MOAT VERDICT: WIDE — 9/10"
SCORE_RE = re.compile(r"MOAT VERDICT:.*?(\d+)\s*/\s*10", re.I)
VERDICT_RE = re.compile(r"MOAT VERDICT:\s*([A-Z][A-Z /-]*?)\s*[—–-]", re.I)


# ---------------------------------------------------------------- LM Studio
def _http_json(url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def detect_lm_studio(base_url):
    """Return (reachable: bool, model_ids: list[str], error: str)."""
    try:
        d = _http_json(base_url.rstrip("/") + "/models", timeout=5)
        ids = [m.get("id", "") for m in d.get("data", [])]
        return True, ids, ""
    except (urllib.error.URLError, OSError, json.JSONDecodeError,
            TimeoutError) as e:
        return False, [], str(e)


def pick_moat_model(model_ids):
    """Prefer a model whose id mentions 'moat' (our quantized student)."""
    for mid in model_ids:
        if "moat" in mid.lower():
            return mid
    return model_ids[0] if model_ids else None


def chat_once(base_url, model, messages, max_tokens=900, timeout=600):
    """One non-streaming chat completion. Appends Qwen3's /no_think switch
    (the student was taught in non-thinking mode; same fix as gen_silver)."""
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = m["content"] + " /no_think"
            break
    d = _http_json(base_url.rstrip("/") + "/chat/completions", {
        "model": model, "messages": msgs,
        "temperature": 0.2, "max_tokens": max_tokens, "stream": False,
    }, timeout=timeout)
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content and msg.get("reasoning_content"):
        raise RuntimeError(
            "Model spent its whole budget 'thinking' and the visible answer "
            "is empty. In LM Studio, click the gear icon next to the loaded "
            "model and switch Reasoning OFF, then retry.")
    return content


# ------------------------------------------------------------- persistence
def load_evals():
    if os.path.exists(EVALS_PATH):
        try:
            with open(EVALS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"evaluations": {}}


def save_evals(evals):
    os.makedirs(os.path.dirname(EVALS_PATH), exist_ok=True)
    tmp = EVALS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(evals, f, indent=1)
    os.replace(tmp, EVALS_PATH)


def evals_dataframe(evals):
    rows = []
    for t, e in sorted((evals.get("evaluations") or {}).items()):
        rows.append({
            "Ticker": t,
            "AI Moat verdict": e.get("verdict") or "?",
            "AI Moat score /10": e.get("score"),
            "Model": e.get("model", ""),
            "Evaluated": (e.get("at") or "")[:19].replace("T", " "),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ prompt
def _load_scan_rows():
    if not os.path.exists(RESULTS_PATH):
        return {}
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        rows = d["results"] if isinstance(d, dict) else d
        return {r["ticker"]: r for r in rows
                if isinstance(r, dict) and r.get("ticker")}
    except (json.JSONDecodeError, OSError, KeyError):
        return {}


def build_messages(ticker, scan_row):
    """EXACT training prompt: rubric system prompt + evidence card."""
    with open(RUBRIC_PATH, encoding="utf-8") as f:
        system = f.read()
    ask = ("Evaluate this company's economic moat using the rubric. "
           "Use the evidence card; follow the output format exactly.\n\n")
    if scan_row:
        try:
            from ai_moat.build_dataset import evidence_block
            ev = evidence_block(scan_row, ticker)
        except Exception:
            ev = f"TICKER: {ticker}\n(evidence card renderer unavailable)"
    else:
        ev = (f"TICKER: {ticker}\n(No scanner evidence card available — "
              f"reason qualitatively from the rubric.)")
    return [{"role": "system", "content": system},
            {"role": "user", "content": ask + ev}], bool(scan_row)


def parse_answer(text):
    """Return (verdict_word or None, score_int or None)."""
    v = VERDICT_RE.search(text)
    s = SCORE_RE.search(text)
    return (v.group(1).strip().upper() if v else None,
            int(s.group(1)) if s else None)


def evaluate_ticker(base_url, model, ticker, scan_row):
    messages, had_card = build_messages(ticker, scan_row)
    answer = chat_once(base_url, model, messages)
    verdict, score = parse_answer(answer)
    return {
        "verdict": verdict, "score": score, "answer": answer,
        "model": model, "had_evidence_card": had_card,
        "at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------- UI
def render():
    st.subheader(tr("🤖 AI Moat Evaluator — your fine-tuned model"))
    st.caption(tr(
        "Runs the moat model YOU trained (Adam Khoo VMI rubric, gold + "
        "contrastive + silver lessons) through LM Studio's local server. "
        "It gets the same rubric + evidence-card prompt it was trained on. "
        "⚠️ AI output — judge it, don't obey it."))

    # ---- connection panel ----------------------------------------------
    c1, c2 = st.columns([3, 1])
    base_url = c1.text_input(
        tr("LM Studio server URL"),
        value=st.session_state.get("moat_ai_url", DEFAULT_BASE_URL),
        help=tr("LM Studio → Developer tab → Start Server. Default port "
                "1234. The DASHBOARD's machine must reach this address — "
                "if the dashboard runs in the cloud sandbox it cannot see "
                "your PC's LM Studio; run locally: "
                "streamlit run scanner/webapp_ui.py"))
    st.session_state["moat_ai_url"] = base_url
    if c2.button(tr("🔌 Detect"), width="stretch"):
        st.session_state.pop("moat_ai_models", None)

    if "moat_ai_models" not in st.session_state:
        ok, ids, err = detect_lm_studio(base_url)
        st.session_state["moat_ai_models"] = (ok, ids, err)
    ok, ids, err = st.session_state["moat_ai_models"]

    if not ok:
        st.warning(tr(
            "LM Studio server not reachable at {u} — {e}\n\n"
            "**Checklist:** 1) LM Studio open → Developer tab → **Start "
            "Server**. 2) Load the moat model INTO THE SERVER (select it "
            "in the Developer tab, not just the Chat tab). 3) If the "
            "dashboard runs in the cloud sandbox, it cannot reach your "
            "PC — run it locally: `streamlit run scanner/webapp_ui.py` "
            "from the repo folder.").replace("{u}", base_url)
            .replace("{e}", err[:120]))
    else:
        auto = pick_moat_model(ids)
        is_ours = auto and "moat" in auto.lower()
        if is_ours:
            st.success(tr("✅ LM Studio connected — YOUR trained moat model "
                          "detected: ") + f"`{auto}`")
        elif auto:
            st.info(tr("LM Studio connected, but no model with 'moat' in "
                       "its name is loaded — using ") + f"`{auto}`" +
                    tr(". To use YOUR trained model: LM Studio → My Models "
                       "→ load the moat-*.gguf into the server."))
        model = st.selectbox(tr("Model"), ids,
                             index=ids.index(auto) if auto in ids else 0)

    evals = load_evals()
    scan_rows = _load_scan_rows()

    # ---- evaluate -------------------------------------------------------
    st.divider()
    e1, e2 = st.columns([3, 2])
    with e1:
        st.markdown(tr("**Evaluate one ticker**"))
        options = sorted(scan_rows.keys())
        pick = st.selectbox(
            tr("Scanned ticker (gets the full evidence card)"),
            [""] + options)
        manual = st.text_input(
            tr("…or any ticker (no evidence card — qualitative only)"),
            placeholder="COST")
        ticker = (manual.strip().upper() or pick or "").strip()
        if st.button(tr("🏰 Evaluate moat"), type="primary",
                     disabled=not (ok and ticker)):
            with st.spinner(tr("Asking your moat model about ") + ticker +
                            tr(" (30-90s on GPU)…")):
                try:
                    res = evaluate_ticker(base_url, model, ticker,
                                          scan_rows.get(ticker))
                    evals["evaluations"][ticker] = res
                    save_evals(evals)
                    st.session_state["moat_ai_last"] = ticker
                except (RuntimeError, urllib.error.URLError, OSError,
                        KeyError, json.JSONDecodeError) as e:
                    st.error(tr("Evaluation failed: ") + str(e)[:400])

    with e2:
        st.markdown(tr("**Batch evaluate**"))
        great = [t for t, r in scan_rows.items() if r.get("is_great")]
        todo = [t for t in great if t not in evals["evaluations"]]
        st.caption(tr("{g} GREAT stocks in last scan · {d} not yet "
                      "evaluated").replace("{g}", str(len(great)))
                   .replace("{d}", str(len(todo))))
        redo = st.checkbox(tr("Re-evaluate already-done tickers"),
                           value=False)
        batch = great if redo else todo
        if st.button(tr("⚡ Evaluate all GREAT stocks"),
                     disabled=not (ok and batch)):
            prog = st.progress(0.0)
            status = st.empty()
            done_n = 0
            for i, t in enumerate(batch):
                status.text(f"{t} ({i + 1}/{len(batch)})")
                try:
                    evals["evaluations"][t] = evaluate_ticker(
                        base_url, model, t, scan_rows.get(t))
                    save_evals(evals)   # checkpoint after every ticker
                    done_n += 1
                except (RuntimeError, urllib.error.URLError, OSError,
                        KeyError, json.JSONDecodeError) as e:
                    st.warning(f"{t}: {str(e)[:160]}")
                prog.progress((i + 1) / len(batch))
            status.text(tr("Done: ") + f"{done_n}/{len(batch)}")

    # ---- last / stored results -----------------------------------------
    last = st.session_state.get("moat_ai_last")
    if last and last in evals["evaluations"]:
        e = evals["evaluations"][last]
        st.divider()
        v, s = e.get("verdict") or "?", e.get("score")
        st.markdown(f"### {last} — {v}" + (f" · **{s}/10**"
                                           if s is not None else ""))
        if not e.get("had_evidence_card"):
            st.caption(tr("⚠️ No scanner evidence card was available — this "
                          "verdict is qualitative-only. Scan the ticker "
                          "first for the full evidence-based answer."))
        st.code(e.get("answer", ""), language=None)

    df = evals_dataframe(evals)
    if not df.empty:
        st.divider()
        h, b = st.columns([4, 1])
        h.markdown(tr("**Saved evaluations**") + f" ({len(df)})" +
                   tr(" — these power the AI-moat filters in the Scanner "
                      "tab"))
        if b.button(tr("🗑️ Clear all"), key="moat_clear"):
            save_evals({"evaluations": {}})
            st.rerun()
        st.dataframe(df, width="stretch", hide_index=True)
        rt = st.selectbox(tr("Re-read a saved answer"),
                          [""] + df["Ticker"].tolist(), key="moat_reread")
        if rt:
            st.code(evals["evaluations"][rt].get("answer", ""),
                    language=None)

    st.divider()
    st.caption(tr("Setup for any user: 1) clone the repo · 2) LM Studio → "
                  "download/import the moat .gguf (needs ~6-8GB VRAM at "
                  "Q5_K_M for the 8B model) · 3) Developer tab → Start "
                  "Server, load the model · 4) streamlit run "
                  "scanner/webapp_ui.py · full guide: "
                  "ai_moat/AI_SUPERGUIDE.md Step 2.5-2.6."))
