"""Download weekly OHLC (adjusted) for all backtest tickers -> weekly_ohlc.csv.

MultiIndex columns (Ticker, Field) with Field in Open/High/Low/Close.
Used only for candlestick charts; the simulation itself keeps using
weekly_adj.csv closes so results are unchanged.
"""
import pandas as pd
import yfinance as yf

TICKERS = ["JNJ", "ABT", "BMY", "BDX", "UNH", "MRK", "PG", "KMB", "CLX",
           "GIS", "HSY", "HRL", "PEP", "CL", "MCD", "GD",
           "TJX", "ROST", "AZO", "ORLY", "NKE", "SYY", "CAH", "ITW", "DHR",
           "LOW", "CVS", "TGT", "DLTR", "CHD", "SBUX", "SYK",
           "PFE", "KO", "GPC", "SPY"]

df = yf.download(TICKERS, start="1999-01-01", end="2014-01-10",
                 interval="1wk", auto_adjust=True, progress=False)
# yfinance returns (Field, Ticker); swap to (Ticker, Field)
df = df[["Open", "High", "Low", "Close"]].swaplevel(axis=1).sort_index(axis=1)
df.to_csv("weekly_ohlc.csv")
print("rows:", len(df), "tickers:", len(df.columns.levels[0]))
print("null Close counts (top 5):")
closes = df.xs("Close", level=1, axis=1)
print(closes.isna().sum().sort_values(ascending=False).head())
