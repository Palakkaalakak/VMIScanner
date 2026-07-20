"""Reliable per-ticker first-trade-date probe using history(period='max')."""
import yfinance as yf, json, time, sys

CANDIDATES = [
    # Mega-cap consumer / healthcare / industrial wide-moats (old-era eligible)
    "KO","PEP","JNJ","PG","MRK","PFE","BMY","ABT","LLY","MCD","XOM","CVX","CAT",
    "GD","CL","KMB","CLX","GIS","HSY","HRL","SYY","MO","DIS","IBM","MMM","GE",
    "BDX","WMT","SPGI","GWW","EMR","ITW","DOV","PH","SWK","PPG","SHW","ADM",
    "K","CPB","CAG","SJM","MKC","TAP","WBA","JCI","HON","BA","LMT","NOC","RTX",
    "TXN","HPQ","MSI","AXP","JPM","BAC","WFC","USB","TRV","AFL","CB","CINF",
    "TGT","LOW","HD","NKE","TJX","ROST","GPS","LB","VFC","SYK","BAX","BDX",
    "MDT","TMO","DHR","CAH","MCK","CVS","AZO","ORLY","GPC","DLTR","DG","CHD",
    "SBUX","MSFT","AAPL","INTC","CSCO","ORCL","ADBE","AMGN","GILD","BIIB",
    "UNH","ANTM","CI","HUM","ADP","PAYX","WM","RSG","UNP","NSC","CSX",
    "AMT","V","MA","GOOGL","AMZN","META","NVDA","CRM","COST","AVGO","ACN",
    "ISRG","REGN","VRTX","NOW","PANW","ODFL","POOL","MNST","LULU","DECK",
]
CANDIDATES = list(dict.fromkeys(CANDIDATES))

out = {}
fails = []
for i, t in enumerate(CANDIDATES):
    try:
        h = yf.Ticker(t).history(period="max", interval="1mo", auto_adjust=True)
        if h is None or h.empty:
            fails.append(t); continue
        out[t] = str(h.index[0].date())
    except Exception as e:
        fails.append(t)
    if (i+1) % 25 == 0:
        print(f"...{i+1}/{len(CANDIDATES)}", flush=True)
    time.sleep(0.15)

json.dump({"first_date": out, "fails": fails}, open("backtest/coverage.json","w"), indent=1)

# Tabulate availability 1 year before each vintage
vintages = [1975,1980,1985,1990,1995,2000,2005,2010,2015,2020]
print("\nVintage availability (data starts >=1y before vintage):")
for v in vintages:
    cutoff = f"{v-1}-06-01"
    avail = sorted([t for t,d in out.items() if d <= cutoff])
    print(f"{v}: {len(avail)} tickers")
    if len(avail) <= 40:
        print("   ", avail)
print("\nFAILS:", fails)
