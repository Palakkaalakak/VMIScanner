"""
i18n for the VMI scanner dashboard — English (default) / Italian toggle.

Design
------
- `tr(s)` looks up the ENGLISH source string in the `IT` dict when
  st.session_state["lang"] == "it". Unknown strings fall back to English,
  so a missing translation can NEVER crash the app or show a raw key.
- The toggle lives at the top of the sidebar in webapp_ui.py
  (key="lang_toggle", default OFF = English).

Terminology rationale (correct Italian financial usage, not literal)
--------------------------------------------------------------------
- "covered call", "PMCC", "LEAPS", "drawdown", "backtest", "roll",
  "strike", "delta" stay in ENGLISH — this is how Italian options
  practitioners actually write and speak (Borsa Italiana / IT trading
  literature keeps these terms untranslated).
- moat            -> "fossato economico"  (Morningstar Italia standard)
- intrinsic value -> "valore intrinseco"
- discount        -> "sconto"
- 200-day SMA     -> "media mobile semplice a 200 giorni"
- equity curve    -> "curva di equity"    (hybrid, standard in IT finance)
- holdings        -> "posizioni"
- trade log       -> "registro delle operazioni"
- compounding     -> "capitalizzazione composta"
- leverage        -> "leva finanziaria"
- GREAT verdict   -> "Eccellente"
- near miss       -> "Quasi idonea"
- NA              -> "N/D" (non disponibile)
"""

import streamlit as st

IT = {
    # ---------- webapp_ui: chrome ----------
    "📈 Great Business Scanner": "📈 Scanner di Aziende Eccellenti",
    "Adam Khoo VMI checklist — data from SEC XBRL / Yahoo / Macrotrends / Finviz. No invented numbers: anything unavailable shows as NA.":
        "Checklist VMI di Adam Khoo — dati da SEC XBRL / Yahoo / Macrotrends / Finviz. Nessun numero inventato: ciò che non è disponibile appare come N/D.",
    "🔎 Scanner": "🔎 Scanner",
    "🕰️ Backtest 2000–2013": "🕰️ Backtest 2000–2013",

    # ---------- webapp_ui: sidebar ----------
    "⚙️ Controls": "⚙️ Controlli",
    "Run a new scan": "Avvia una nuova scansione",
    "Universe": "Universo",
    "Include Nasdaq-100": "Includi Nasdaq-100",
    "Include curated extras": "Includi selezione extra curata",
    "Workers": "Processi paralleli",
    "▶ Run full scan": "▶ Avvia scansione completa",
    "Scan a single ticker": "Analizza un singolo titolo",
    "Ticker": "Titolo",
    "🔍 Scan ticker": "🔍 Analizza titolo",
    "Scanning…": "Scansione in corso…",
    "Scan running in background — refresh to see progress.":
        "Scansione in esecuzione in background — aggiorna la pagina per vedere i progressi.",

    # ---------- webapp_ui: results header / metrics ----------
    "Ad-hoc result": "Risultato singolo",
    "Dismiss": "Chiudi",
    "Universe size": "Dimensione universo",
    "Excluded": "Esclusi",
    "Deep-scanned": "Analizzati a fondo",
    "GREAT": "Eccellenti",
    "Near misses": "Quasi idonee",

    # ---------- webapp_ui: filters ----------
    "Verdict": "Giudizio",
    "All": "Tutti",
    "GREAT only": "Solo Eccellenti",
    "Near miss (1 fail)": "Quasi idonea (1 criterio mancato)",
    "Sector": "Settore",
    "Search ticker / company": "Cerca titolo / società",
    "Custom filters": "Filtri personalizzati",
    "Sort by": "Ordina per",
    "(none)": "(nessuno)",
    "Ascending": "Crescente",
    "Descending": "Decrescente",
    "Direction": "Direzione",
    "Min": "Min",
    "Max": "Max",

    # ---------- webapp_ui: detail ----------
    "Check detail": "Dettaglio dei criteri",
    "Pick a ticker": "Scegli un titolo",
    "Price": "Prezzo",
    "Intrinsic value (DCF)": "Valore intrinseco (DCF)",
    "Discount": "Sconto",
    "DCF growth used": "Crescita usata nel DCF",
    "NA = data not available from any free source (never invented).":
        "N/D = dato non disponibile da alcuna fonte gratuita (mai inventato).",
    "No results match the current filters.": "Nessun risultato corrisponde ai filtri attuali.",
    "No scan results yet — run a scan from the sidebar.":
        "Ancora nessun risultato — avvia una scansione dalla barra laterale.",

    # ---------- backtest_tab: header ----------
    "## 🕰️ VMI backtest — day-by-day reality simulation":
        "## 🕰️ Backtest VMI — simulazione giorno per giorno della realtà",
    "Rules locked before the test. No hindsight. Benchmark: SPY total return.":
        "Regole fissate prima del test. Nessun senno di poi. Benchmark: rendimento totale SPY.",

    # ---------- backtest_tab: sub-tabs ----------
    "📈 Equity curves": "📈 Curve di equity",
    "🕯️ Candles": "🕯️ Grafico a candele",
    "📅 Year by year": "📅 Anno per anno",
    "📊 Holdings": "📊 Posizioni",
    "📜 Trade log": "📜 Registro delle operazioni",
    "📉 Drawdowns": "📉 Drawdown",
    "🧮 Stats": "🧮 Statistiche",
    "🎯 Options overlays": "🎯 Strategie in opzioni",
    "💼 Today's portfolio": "💼 Portafoglio di oggi",
    "📖 Method & caveats": "📖 Metodo e avvertenze",

    # ---------- backtest_tab: equity curve controls ----------
    "Log scale": "Scala logaritmica",
    "Normalize to $100": "Normalizza a 100 $",
    "Show SPY": "Mostra SPY",
    "Show recessions": "Mostra recessioni",
    "Portfolio value ($)": "Valore del portafoglio ($)",
    "Hover the chart for exact values.": "Passa il mouse sul grafico per i valori esatti.",
    "Static charts": "Grafici statici",
    "Weekly candles of the account value — 40-week SMA in orange.":
        "Candele settimanali del valore del conto — media mobile a 40 settimane in arancione.",

    # ---------- backtest_tab: year by year ----------
    "Show table": "Mostra tabella",
    "Years beating SPY": "Anni sopra SPY",
    "Best year": "Anno migliore",
    "Worst year": "Anno peggiore",
    "Annual returns vs SPY": "Rendimenti annui vs SPY",

    # ---------- backtest_tab: holdings / log ----------
    "Account": "Conto",
    "Pick a holding": "Scegli una posizione",
    "Green = buy tranches. Orange line = 200-day SMA.":
        "Verde = tranche di acquisto. Linea arancione = media mobile semplice a 200 giorni.",
    "Trades": "Operazioni",
    "Buys": "Acquisti",
    "Sells": "Vendite",
    "Tickers traded": "Titoli negoziati",
    "pct_of_iv = price paid as % of intrinsic value at purchase.":
        "pct_of_iv = prezzo pagato in % del valore intrinseco al momento dell'acquisto.",
    "Corrections shaded. Depth measured close-to-close.":
        "Correzioni evidenziate. Profondità misurata da chiusura a chiusura.",
    "Underwater = % below the previous all-time high.":
        "Underwater = % sotto il massimo storico precedente.",

    # ---------- backtest_tab: labels ----------
    "Bedrock tier": "Livello del bedrock",
    "Strategy": "Strategia",
    "📈 Plain shares": "📈 Solo azioni",
    "Proj. EPS growth %": "Crescita EPS attesa %",
    "Price $": "Prezzo $",
    "Intrinsic value $": "Valore intrinseco $",
    "Discount %": "Sconto %",
    "Moat (manually verified)": "Fossato economico (verifica manuale)",

    # ---------- NICE account labels ----------
    "🛡️ Defensive": "🛡️ Difensivo",
    "🚀 Growth": "🚀 Crescita",
    "S&P 500 (SPY)": "S&P 500 (SPY)",
    "🛡️ Defensive + covered calls": "🛡️ Difensivo + covered call",
    "🚀 Growth + covered calls": "🚀 Crescita + covered call",
    "🛡️ Defensive + PMCC": "🛡️ Difensivo + PMCC",
    "🚀 Growth + PMCC": "🚀 Crescita + PMCC",

    # ---------- shared ----------
    "NA": "N/D",
    "Company": "Società",
    "Fails": "Criteri mancati",
    "Warnings": "Avvertimenti",
    "Data source": "Fonte dati",
    "Gross margin %": "Margine lordo %",
}


def tr(s: str) -> str:
    """Translate a UI string when the Italian toggle is on."""
    if st.session_state.get("lang", "en") == "it":
        return IT.get(s, s)
    return s
