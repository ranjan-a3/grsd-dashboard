import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional

INDIA_SECTORS = {
    "IT":     {"label": "Nifty IT",     "index": "^CNXIT",     "stocks": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS"]},
    "BANK":   {"label": "Bank Nifty",   "index": "^NSEBANK",   "stocks": ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS","FEDERALBNK.NS"]},
    "AUTO":   {"label": "Nifty Auto",   "index": "^CNXAUTO",   "stocks": ["MARUTI.NS","TATAMOTORS.NS","HEROMOTOCO.NS","EICHERMOT.NS","ASHOKLEY.NS","MOTHERSON.NS","BALKRISIND.NS","MRF.NS"]},
    "PHARMA": {"label": "Nifty Pharma", "index": "^CNXPHARMA", "stocks": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","LUPIN.NS","BIOCON.NS","AUROPHARMA.NS","ALKEM.NS"]},
    "FMCG":   {"label": "Nifty FMCG",  "index": "^CNXFMCG",   "stocks": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS"]},
    "METAL":  {"label": "Nifty Metal",  "index": "^CNXMETAL",  "stocks": ["TATASTEEL.NS","HINDALCO.NS","JSWSTEEL.NS","SAIL.NS","VEDL.NS","NMDC.NS","NATIONALUM.NS","HINDCOPPER.NS"]},
    "ENERGY": {"label": "Nifty Energy", "index": "^CNXENERGY", "stocks": ["RELIANCE.NS","ONGC.NS","NTPC.NS","POWERGRID.NS","COALINDIA.NS","BPCL.NS","IOC.NS","GAIL.NS"]},
    "REALTY": {"label": "Nifty Realty", "index": "^CNXREALTY", "stocks": ["DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS","SOBHA.NS","BRIGADE.NS","PHOENIXLTD.NS","MACROTECH.NS"]},
    "INFRA":  {"label": "Nifty Infra",  "index": "^CNXINFRA",  "stocks": ["LT.NS","ULTRACEMCO.NS","ADANIPORTS.NS","SIEMENS.NS","ABB.NS","BHEL.NS","IRFC.NS","RVNL.NS"]},
    "MEDIA":  {"label": "Nifty Media",  "index": "^CNXMEDIA",  "stocks": ["ZEEL.NS","SUNTV.NS","PVRINOX.NS","NETWORK18.NS","DISHTV.NS","NAZARA.NS"]},
}
US_SECTORS = {
    "XLK":  {"label": "Technology",        "stocks": ["AAPL","MSFT","NVDA","AVGO","CRM","AMD","INTC","QCOM","TXN","NOW"]},
    "XLF":  {"label": "Financials",        "stocks": ["JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","C","COF"]},
    "XLE":  {"label": "Energy",            "stocks": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","OXY","HES","BKR"]},
    "XLV":  {"label": "Healthcare",        "stocks": ["JNJ","UNH","LLY","ABT","MRK","TMO","DHR","BMY","AMGN","ISRG"]},
    "XLY":  {"label": "Cons Discretionary","stocks": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","BKNG","ORLY","GM"]},
    "XLP":  {"label": "Cons Staples",      "stocks": ["PG","KO","PEP","WMT","COST","PM","CL","GIS","MO","KHC"]},
    "XLI":  {"label": "Industrials",       "stocks": ["GE","RTX","CAT","HON","UPS","DE","LMT","BA","MMM","ETN"]},
    "XLB":  {"label": "Materials",         "stocks": ["LIN","APD","ECL","DD","NEM","FCX","ALB","CE","PPG","IFF"]},
    "XLRE": {"label": "Real Estate",       "stocks": ["PLD","AMT","EQIX","SPG","O","DLR","PSA","EXR","WELL","AVB"]},
    "XLU":  {"label": "Utilities",         "stocks": ["NEE","DUK","SO","AEP","EXC","SRE","XEL","ED","D","PCG"]},
    "XLC":  {"label": "Communication",     "stocks": ["META","GOOGL","VZ","T","NFLX","DIS","TMUS","EA","PARA","CMCSA"]},
}
INDIA_BENCHMARK = "^NSEI"
US_BENCHMARK = "SPY"

def _flat(df):
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def get_close(df):
    df = _flat(df)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_single(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        df = _flat(df)
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(tickers, period="1y", interval="1d"):
    result = {}
    for ticker in tickers:
        ticker = ticker.strip()
        if not ticker:
            continue
        df = fetch_single(ticker, period, interval)
        if not df.empty:
            result[ticker] = df
    return result

@st.cache_data(ttl=900, show_spinner=False)
def fetch_sector_indices(market="INDIA"):
    if market == "INDIA":
        ticker_map = {k: v["index"] for k, v in INDIA_SECTORS.items()}
    else:
        ticker_map = {k: k for k in US_SECTORS}
    result = {}
    for key, ticker in ticker_map.items():
        df = fetch_single(ticker, period="1y")
        if not df.empty:
            result[key] = df
    return result

@st.cache_data(ttl=900, show_spinner=False)
def fetch_sector_stocks(sector_key, market="INDIA"):
    if market == "INDIA":
        stocks = INDIA_SECTORS.get(sector_key, {}).get("stocks", [])
    else:
        stocks = US_SECTORS.get(sector_key, {}).get("stocks", [])
    return fetch_ohlcv(stocks, period="1y")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_full_universe(market="INDIA"):
    sectors = INDIA_SECTORS if market == "INDIA" else US_SECTORS
    return {sk: fetch_sector_stocks(sk, market) for sk in sectors}

def compute_relative_perf(sector_data, benchmark_df, lookbacks):
    benchmark_df = _flat(benchmark_df)
    if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
        return pd.DataFrame()
    bench = benchmark_df["Close"].rename("bench")
    rows = []
    for key, df in sector_data.items():
        df = _flat(df)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        aligned = pd.concat([df["Close"], bench], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            continue
        row = {"sector": key}
        for label, lb in lookbacks.items():
            n = min(lb, len(aligned) - 1)
            if n < 1:
                row[label] = 0.0
                continue
            s = (aligned["Close"].iloc[-1] / aligned["Close"].iloc[-n] - 1) * 100
            b = (aligned["bench"].iloc[-1] / aligned["bench"].iloc[-n] - 1) * 100
            row[label] = round(s - b, 2)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("sector")

def estimate_delivery_pct(df):
    df = _flat(df)
    if df is None or df.empty or "High" not in df.columns:
        return 50.0
    last = df.iloc[-1]
    hl = float(last["High"]) - float(last["Low"])
    if hl == 0:
        return 50.0
    return round(abs(float(last["Close"]) - float(last["Open"])) / hl * 100, 1)

def sector_label(key, market):
    if market == "INDIA":
        return INDIA_SECTORS.get(key, {}).get("label", key)
    return US_SECTORS.get(key, {}).get("label", key)

def all_sector_labels(market):
    if market == "INDIA":
        return {k: v["label"] for k, v in INDIA_SECTORS.items()}
    return {k: v["label"] for k, v in US_SECTORS.items()}
