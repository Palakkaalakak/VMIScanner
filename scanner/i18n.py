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
  "strike", "delta", "buy-and-hold" stay in ENGLISH — this is how Italian
  options practitioners actually write and speak (Borsa Italiana / IT
  trading literature keeps these terms untranslated).
- moat            -> "fossato economico"  (Morningstar Italia standard)
- intrinsic value -> "valore intrinseco"
- discount        -> "sconto"
- 200-day SMA     -> "media mobile semplice a 200 giorni" / "SMA a 40 settimane"
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
    "📈 VMI Great Business Scanner": "📈 Scanner VMI di Aziende Eccellenti",
    "S&P 500 + Dow Jones 30 (toggle) · fundamentals-only checklist · SEC Company Facts primary, "
    "Yahoo + macrotrends fallbacks · trend/average checks require the "
    "full 20y window by default (toggle allows 20/15/10y any-pass) · "
    "IV = v13 sector-calibrated 20y DCF (StockOracle-matched: per-sector "
    "base-flow blend + fitted growth model, CAPM discount, + net cash; "
    "no terminal value · 36/36 calibration tickers within ±7%)":
        "S&P 500 + Dow Jones 30 (attivabile) · checklist basata solo sui fondamentali · fonte primaria SEC Company Facts, "
        "con fallback Yahoo + Macrotrends · i controlli su trend/medie richiedono di default "
        "l'intera finestra di 20 anni (l'opzione consente il superamento su 20/15/10 anni) · "
        "VI = DCF a 20 anni v13 calibrato per settore (allineato a StockOracle: mix di flussi base "
        "per settore + modello di crescita stimato, tasso di sconto CAPM, + cassa netta; "
        "senza valore terminale · 36/36 titoli di calibrazione entro ±7%)",
    "🔎 Scanner": "🔎 Scanner",
    "🕰️ Backtest 2000–2013": "🕰️ Backtest 2000–2013",
    "🇮🇹 Italiano": "🇮🇹 Italiano",

    # ---------- webapp_ui: sidebar ----------
    "Run scanner": "Avvia lo scanner",
    "Allow 20/15/10y any-pass": "Consenti superamento su 20/15/10 anni",
    "Accept 5y-only passes": "Accetta superamenti solo su 5 anni",
    "Include Dow Jones 30": "Includi Dow Jones 30",
    "Force fresh data (ignore cache)": "Forza dati aggiornati (ignora la cache)",
    "Re-score all tickers (no resume)": "Rivaluta tutti i titoli (senza ripresa)",
    "Scan specific tickers": "Analizza titoli specifici",
    "Comma-separated tickers": "Titoli separati da virgola",
    "Scan tickers": "Analizza titoli",
    "Ticker": "Titolo",

    # ---------- webapp_ui: metrics ----------
    "✅ Great": "✅ Eccellenti",
    "🟡 Near miss (1 fail)": "🟡 Quasi idonee (1 criterio mancato)",
    "❌ Failed": "❌ Bocciate",
    "Errors": "Errori",
    "Excluded": "Escluse",

    # ---------- webapp_ui: filters / table ----------
    "Verdict": "Giudizio",
    "Sector": "Settore",
    "Search ticker / company": "Cerca titolo / società",
    "🔧 Custom filters & sorting (all known data)": "🔧 Filtri personalizzati e ordinamento (tutti i dati noti)",
    "Stack any number of numeric filters on top of the verdict "
    "filter above. Blank rows in the data (NA) are **kept** by default — "
    "NA never disqualifies — untick to drop them.":
        "Sovrapponi un numero qualsiasi di filtri numerici al filtro sul giudizio qui sopra. "
        "Le righe con dato mancante (N/D) sono **mantenute** di default — "
        "N/D non squalifica mai — deseleziona per escluderle.",
    "Number of custom filters": "Numero di filtri personalizzati",
    "Keep NA": "Mantieni N/D",
    "Sort by": "Ordina per",
    "(none)": "(nessuno)",
    "Ascending": "Crescente",
    "Descending": "Decrescente",
    "Show ALL data columns in the table (CAGRs, projections, …)":
        "Mostra TUTTE le colonne di dati nella tabella (CAGR, proiezioni, …)",

    # ---------- webapp_ui: detail ----------
    "Check detail": "Dettaglio dei criteri",
    "Price": "Prezzo",
    "Intrinsic value (DCF)": "Valore intrinseco (DCF)",
    "Discount": "Sconto",
    "DCF growth used": "Crescita usata nel DCF",
    "NA = data not reported by the source or check not applicable "
    "to this company type — NA never disqualifies a stock.":
        "N/D = dato non pubblicato dalla fonte o controllo non applicabile "
        "a questo tipo di società — N/D non squalifica mai un titolo.",
    "No results yet — hit **Run full scan** in the sidebar "
    "(S&P 500 + Dow 30 by default).":
        "Ancora nessun risultato — premi **Avvia scansione completa** nella barra laterale "
        "(S&P 500 + Dow 30 di default).",

    # ---------- backtest_tab: sub-tabs ----------
    "📈 Equity curves": "📈 Curve di equity",
    "🕯️ Candles": "🕯️ Grafico a candele",
    "📊 Year by year": "📊 Anno per anno",
    "🧱 Holdings & contribution": "🧱 Posizioni e contributo",
    "🔍 Stock charts": "🔍 Grafici dei titoli",
    "📜 Trade log": "📜 Registro delle operazioni",
    "🌊 Corrections": "🌊 Correzioni",
    "🎯 Options overlays": "🎯 Strategie in opzioni",
    "🛒 Today's portfolio (2026)": "🛒 Portafoglio di oggi (2026)",
    "🧪 Method & caveats": "🧪 Metodo e avvertenze",

    # ---------- backtest_tab: header prose ----------
    "Two $1M accounts, 16 wide-moat non-dotcom businesses each, **DCF-gated entries "
    "only** (course-style 20y DCF, rf 6.5%, β×4% MRP), tranche adds only at the "
    "40-week SMA while under IV, sells **only** on fraud/scandal (BMY→PFE, UNH→KO, "
    "CAH→GPC). The single allowed piece of hindsight: a defensive sector tilt for "
    "the lost decade. Dividends reinvested via adjusted prices.":
        "Due conti da 1 M$, 16 aziende con ampio fossato economico (non dot-com) ciascuno, "
        "**ingressi consentiti solo sotto il valore intrinseco DCF** (DCF a 20 anni in stile corso, "
        "tasso privo di rischio 6,5%, premio per il rischio β×4%), incrementi a tranche solo sulla "
        "SMA a 40 settimane finché il prezzo resta sotto il VI, vendite **solo** in caso di frode/scandalo "
        "(BMY→PFE, UNH→KO, CAH→GPC). L'unico senno di poi ammesso: un orientamento difensivo di settore "
        "per il decennio perduto. Dividendi reinvestiti tramite prezzi rettificati.",

    # ---------- backtest_tab: equity curve controls ----------
    "Log scale": "Scala logaritmica",
    "Normalize to 1.0": "Normalizza a 1,0",
    "Include covered-call variants": "Includi le varianti con covered call",
    "Hover for values · drag to zoom · double-click to reset.":
        "Passa il mouse per i valori · trascina per ingrandire · doppio clic per ripristinare.",
    "Static charts with correction shading": "Grafici statici con correzioni evidenziate",

    # ---------- backtest_tab: candles ----------
    "Portfolio equity resampled into **monthly OHLC candles** — green/red months "
    "show the path, not just the endpoint.":
        "Equity del portafoglio ricampionata in **candele mensili OHLC** — i mesi verdi/rossi "
        "mostrano il percorso, non solo il punto di arrivo.",

    # ---------- backtest_tab: year by year ----------
    "Years defensive beat SPY": "Anni in cui il Difensivo batte SPY",
    "Years growth beat SPY": "Anni in cui il Crescita batte SPY",
    "Negative years (G / D / SPY)": "Anni negativi (C / D / SPY)",

    # ---------- backtest_tab: holdings / stock charts ----------
    "Account": "Conto",
    "Pick a holding": "Scegli una posizione",
    "Monthly candles · orange = 40-week SMA · purple dashed = DCF intrinsic value · "
    "▲ green = buy/add at support under IV · ▼ red = scandal sell · blue ▲ = "
    "replacement buy · bottom panel = your position value.":
        "Candele mensili · arancione = SMA a 40 settimane · viola tratteggiato = valore intrinseco DCF · "
        "▲ verde = acquisto/incremento sul supporto sotto il VI · ▼ rosso = vendita per scandalo · ▲ blu = "
        "acquisto sostitutivo · pannello inferiore = valore della tua posizione.",
    "No per-stock charts found — run backtest/plot_suite.py.":
        "Nessun grafico per singolo titolo trovato — esegui backtest/plot_suite.py.",

    # ---------- backtest_tab: trade log ----------
    "Trades": "Operazioni",
    "Initial buys": "Acquisti iniziali",
    "Support adds": "Incrementi sul supporto",
    "Scandal sells": "Vendite per scandalo",
    "Total capital deployed": "Capitale totale impiegato",
    "`pct_of_iv` = purchase price as % of that day's DCF intrinsic value — "
    "every entry was made below IV.":
        "`pct_of_iv` = prezzo di acquisto in % del valore intrinseco DCF di quel giorno — "
        "ogni ingresso è stato effettuato sotto il VI.",

    # ---------- backtest_tab: corrections ----------
    "Every peak-to-trough decline of **7%+**, with how long the fall took "
    "and how long recovery took.":
        "Ogni ribasso dal massimo al minimo di **almeno il 7%**, con la durata della discesa "
        "e il tempo di recupero.",

    # ---------- backtest_tab: options overlays ----------
    "**Covered-call variant note:** the CC curves are a day-by-day simulation on "
    "unadjusted prices with dividends credited as cash and Black-Scholes call "
    "pricing at each stock's realized volatility, while the base curves use "
    "adjusted prices (dividends auto-reinvested). Same economics, slightly "
    "different bookkeeping — so compare each CC curve to the index and to its own "
    "start, not penny-for-penny against its base twin. Growth book calls were "
    "sold on: ":
        "**Nota sulla variante covered call:** le curve CC sono una simulazione giorno per giorno "
        "su prezzi non rettificati, con dividendi accreditati in contanti e prezzatura delle call "
        "con Black-Scholes alla volatilità realizzata di ciascun titolo, mentre le curve base usano "
        "prezzi rettificati (dividendi reinvestiti automaticamente). Stessa economia, contabilità "
        "leggermente diversa — quindi confronta ogni curva CC con l'indice e con il proprio punto di "
        "partenza, non centesimo per centesimo con la gemella base. Nel conto Crescita le call sono "
        "state vendute su: ",
    "**Verdict: structural.** Every overlay beats its own buy-and-hold book in "
    "BOTH sub-periods, not just the lost decade. The edge shrinks in the trending "
    "2013–26 leg (covered calls: ~+12pp → ~+5pp over plain) exactly as theory "
    "predicts — call-selling pays best sideways — but it never flips negative. "
    "Sub-period CAGRs use the same single continuous run split at 2013-12-27.":
        "**Verdetto: strutturale.** Ogni strategia in opzioni batte il proprio conto buy-and-hold in "
        "ENTRAMBI i sotto-periodi, non solo nel decennio perduto. Il vantaggio si riduce nella fase "
        "in trend 2013–26 (covered call: da ~+12 p.p. a ~+5 p.p. rispetto alle sole azioni), esattamente "
        "come prevede la teoria — la vendita di call rende di più nei mercati laterali — ma non diventa "
        "mai negativo. I CAGR dei sotto-periodi usano la stessa singola simulazione continua divisa al 27-12-2013.",
    "The pattern repeats in every era on these value books — more calls, more "
    "return — because IV-gated entries buy cheap, rarely-runaway names. The "
    "cutoff matters most when the book holds true hyper-growers (SBUX/DLTR/ORLY "
    "at 22–27% growth).":
        "Lo schema si ripete in ogni epoca su questi conti value — più call, più rendimento — "
        "perché gli ingressi vincolati al valore intrinseco comprano titoli a sconto che raramente "
        "corrono via. La soglia conta soprattutto quando il conto detiene veri iper-compounder "
        "(SBUX/DLTR/ORLY con crescita del 22–27%).",
    "Overlay: sell covered calls on the CC-viable names":
        "Strategia: vendita di covered call sui titoli idonei alle CC",
    "Overlay: PMCC (deep-ITM long calls instead of shares, leveraged)":
        "Strategia: PMCC (call lunghe deep-ITM al posto delle azioni, con leva finanziaria)",
    "PMCC variant (same variant on both books)":
        "Variante PMCC (stessa variante su entrambi i conti)",
    "Full pyramid (max leverage)": "Piramide completa (leva massima)",
    "Half-pyramid": "Mezza piramide",
    "Convert→shares (survivable)": "Conversione→azioni (rischio sostenibile)",
    "Both books are drawn with the SAME PMCC variant "
    "so the chart matches the options-tab table "
    "apples-to-apples.":
        "Entrambi i conti sono tracciati con la STESSA variante PMCC, "
        "così il grafico corrisponde alla tabella della scheda opzioni "
        "in un confronto omogeneo.",
    "Growth book, 2000–2013. The defensive book is insensitive — all 16 names "
    "are already ≤ 15%.":
        "Conto Crescita, 2000–2013. Il conto Difensivo non ne risente — tutti i 16 titoli "
        "sono già ≤ 15%.",
    "Growth-book names that got calls: ": "Titoli del conto Crescita su cui sono state vendute call: ",
    "Run `python backtest/cc_2000_2013.py` and `python backtest/options_2000_2013.py` "
    "to generate the options-overlay artifacts.":
        "Esegui `python backtest/cc_2000_2013.py` e `python backtest/options_2000_2013.py` "
        "per generare i risultati delle strategie in opzioni.",

    # ---------- backtest_tab: today's portfolio ----------
    "Workflow: scanner DCF first → manual moat verification second. Bottom-heavy "
    "on the bedrock pyramid — Healthcare (5) › Financial toll booths (3) › "
    "SaaS/vertical software (3) › Comms (2) › Consumer compounders (2) › "
    "Industrial services (1) › Materials (0 — LIN/ECL/SHW at fair value, Tier-3 "
    "'wait for the discount'). No commodity knife-fighters (INTC/AMD class "
    "rejected — no sustainable advantage). g ≤ 15% gets the PMCC overlay; faster "
    "growers stay plain shares.":
        "Flusso di lavoro: prima il DCF dello scanner → poi la verifica manuale del fossato economico. "
        "Piramide del bedrock con base pesante — Sanità (5) › Caselli finanziari (3) › "
        "SaaS/software verticale (3) › Comunicazioni (2) › Compounder dei consumi (2) › "
        "Servizi industriali (1) › Materiali (0 — LIN/ECL/SHW a fair value, Livello 3: "
        "'aspetta lo sconto'). Nessun combattente su commodity (classe INTC/AMD respinta — "
        "nessun vantaggio sostenibile). g ≤ 15% riceve la strategia PMCC; le società a crescita "
        "più rapida restano in sole azioni.",
    "Run the portfolio selection to generate `backtest/portfolio_2026.json`.":
        "Esegui la selezione del portafoglio per generare `backtest/portfolio_2026.json`.",
    "Bedrock tier": "Livello del bedrock",
    "Strategy": "Strategia",
    "📈 Plain shares": "📈 Solo azioni",
    "Proj. EPS growth %": "Crescita EPS attesa %",
    "Price $": "Prezzo $",
    "Intrinsic value $": "Valore intrinseco $",
    "Discount %": "Sconto %",
    "Moat (manually verified)": "Fossato economico (verifica manuale)",

    # ---------- backtest_tab: method ----------
    "Artifacts: `backtest/simulate2.py` (engine) · `trades.json` (structured log) · "
    "`stats2.json` · `charts/` (48 charts) — all committed to the repo.":
        "File prodotti: `backtest/simulate2.py` (motore) · `trades.json` (registro strutturato) · "
        "`stats2.json` · `charts/` (48 grafici) — tutti salvati nel repository.",
    "No backtest artifacts found in `backtest/` — run "
    "`python backtest/simulate2.py` then `python backtest/plot_suite.py`.":
        "Nessun risultato di backtest trovato in `backtest/` — esegui "
        "`python backtest/simulate2.py` e poi `python backtest/plot_suite.py`.",

    # ---------- NICE account labels ----------
    "🛡️ Defensive": "🛡️ Difensivo",
    "🚀 Growth": "🚀 Crescita",
    "S&P 500 (SPY)": "S&P 500 (SPY)",
    "🛡️ Defensive + covered calls": "🛡️ Difensivo + covered call",
    "🚀 Growth + covered calls": "🚀 Crescita + covered call",
    "🛡️ Defensive + PMCC": "🛡️ Difensivo + PMCC",
    "🚀 Growth + PMCC": "🚀 Crescita + PMCC",

    # ---------- table headers (options tab, sortable numeric tables) ----------
    "Book": "Conto",
    "Plain CAGR %": "CAGR solo azioni %",
    "With CC CAGR %": "CAGR con CC %",
    "Plain final $": "Finale solo azioni $",
    "With CC final $": "Finale con CC $",
    "CC max DD %": "Max drawdown CC %",
    "Calls on": "Call vendute su",
    "Cutoff": "Soglia",
    "Names with calls": "Titoli con call",
    "CAGR %": "CAGR %",
    "Max DD %": "Max drawdown %",
    "Start": "Inizio",
    "Variant": "Variante",
    "Final $": "Valore finale $",
    "CAGR 2000–26 %": "CAGR 2000–26 %",
    "CAGR 00–13 %": "CAGR 00–13 %",
    "CAGR 13–26 %": "CAGR 13–26 %",
    "Final ($1M start)": "Finale (partenza 1 M$)",
    "buy & hold": "buy & hold",
    "covered calls": "covered call",
    "PMCC→shares": "PMCC→azioni",
    "PMCC half-pyramid": "PMCC mezza piramide",
    "PMCC full pyramid": "PMCC piramide completa",
    "full pyramid": "piramide completa",
    "half-pyramid": "mezza piramide",
    "convert→shares": "conversione→azioni",
    "Direction": "Direzione",
    "{n} stocks shown · click any column header to sort "
    "ascending/descending · Discount % > 0 = below intrinsic "
    "value, < 0 = premium":
        "{n} titoli mostrati · clicca l'intestazione di una colonna per "
        "ordinare in modo crescente/decrescente · Sconto % > 0 = sotto il "
        "valore intrinseco, < 0 = a premio",

    # ---------- footers / disclaimers ----------
    "⚠️ Educational research only — not investment advice. "
    "Backtest option prices are Black-Scholes estimates (real "
    "options history does not exist back to 2000); past "
    "performance does not guarantee future results.":
        "⚠️ Solo ricerca a scopo didattico — non è consulenza finanziaria. "
        "I prezzi delle opzioni nel backtest sono stime Black-Scholes (lo "
        "storico reale delle opzioni non esiste fino al 2000); i rendimenti "
        "passati non garantiscono risultati futuri.",
    "⚠️ Educational research only — not investment advice. "
    "Data from free public sources (SEC XBRL, Yahoo, "
    "Macrotrends, Finviz) and may contain errors; verify "
    "before trading. DCF values are model outputs, not price "
    "targets.":
        "⚠️ Solo ricerca a scopo didattico — non è consulenza finanziaria. "
        "Dati da fonti pubbliche gratuite (SEC XBRL, Yahoo, Macrotrends, "
        "Finviz), possono contenere errori; verifica prima di operare. "
        "I valori DCF sono output di un modello, non target di prezzo.",

    # ---------- shared ----------
    "Dismiss": "Chiudi",
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
