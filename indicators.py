import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


# ─── Column safety helper ────────────────────────────────────────────────────
# New yfinance returns MultiLevel columns. This safely extracts any column.

def _col(df, name):
    """Safely get a column from df whether columns are flat or MultiLevel."""
    if isinstance(df.columns, pd.MultiIndex):
        matches = [c for c in df.columns if c[0] == name]
        if matches:
            return df[matches[0]]
        return pd.Series(dtype=float)
    if name in df.columns:
        return df[name]
    return pd.Series(dtype=float)


def _flat(df):
    """Flatten MultiLevel columns to single level."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df


# ─── Color maps ──────────────────────────────────────────────────────────────

RRG_COLORS = {"Leading": "#00c853", "Weakening": "#ff9800",
              "Lagging": "#ef5350", "Improving": "#2196f3"}

STAGE_COLORS = RRG_COLORS


# ─── Moving averages ─────────────────────────────────────────────────────────

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def sma(series, period):
    return series.rolling(window=period, min_periods=1).mean()


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(df, period=14):
    df = _flat(df)
    H = df["High"]
    L = df["Low"]
    C = df["Close"]
    prev_c = C.shift(1)
    tr = pd.concat([H - L, (H - prev_c).abs(), (L - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─── ADX ─────────────────────────────────────────────────────────────────────

def adx(df, period=14):
    df = _flat(df)
    H = df["High"]
    L = df["Low"]
    C = df["Close"]
    prev_H = H.shift(1)
    prev_L = L.shift(1)
    prev_C = C.shift(1)
    tr = pd.concat([H - L, (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(axis=1)
    dm_p = pd.Series(np.where((H - prev_H) > (prev_L - L), np.maximum(H - prev_H, 0), 0), index=df.index)
    dm_m = pd.Series(np.where((prev_L - L) > (H - prev_H), np.maximum(prev_L - L, 0), 0), index=df.index)
    atr_ = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr_.replace(0, np.nan)
    di_m = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx_v = dx.ewm(span=period, adjust=False).mean()
    return pd.DataFrame({"ADX": adx_v, "+DI": di_p, "-DI": di_m}, index=df.index)


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── MACD ────────────────────────────────────────────────────────────────────

def macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"MACD": line, "Signal": sig, "Histogram": hist}, index=series.index)


# ─── Bollinger Bands ─────────────────────────────────────────────────────────

def bollinger_bands(series, period=20, std_dev=2.0):
    mid = sma(series, period)
    std = series.rolling(period).std()
    return pd.DataFrame({"Upper": mid + std_dev * std, "Mid": mid, "Lower": mid - std_dev * std}, index=series.index)


# ─── SMI ─────────────────────────────────────────────────────────────────────

def smi(df, period=14, sm1=3, sm2=3, sig=8):
    df = _flat(df)
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    mid = (hi + lo) / 2
    diff = df["Close"] - mid
    num = diff.ewm(span=sm1, adjust=False).mean().ewm(span=sm2, adjust=False).mean()
    den = (hi - lo).div(2).ewm(span=sm1, adjust=False).mean().ewm(span=sm2, adjust=False).mean()
    val = 100 * num / den.replace(0, np.nan)
    sig_ = val.ewm(span=sig, adjust=False).mean()
    return pd.DataFrame({"SMI": val, "SMI_Signal": sig_}, index=df.index)


# ─── Relative Strength ───────────────────────────────────────────────────────

def rs_ratio_series(stock, bench):
    aligned = pd.concat([stock.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
    rs = aligned["s"] / aligned["b"]
    return (rs / rs.iloc[0] * 100)


def rs_score(stock, bench, period=63):
    aligned = pd.concat([stock.rename("s"), bench.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    n = min(period, len(aligned) - 1)
    s = (aligned["s"].iloc[-1] / aligned["s"].iloc[-n] - 1) * 100
    b = (aligned["b"].iloc[-1] / aligned["b"].iloc[-n] - 1) * 100
    return round(s - b, 2)


# ─── Pivot S/R ───────────────────────────────────────────────────────────────

def pivot_levels(df):
    df = _flat(df)
    if df.empty or len(df) < 2:
        return {}
    row = df.iloc[-2]
    H = float(row["High"])
    L = float(row["Low"])
    C = float(row["Close"])
    P = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {k: round(v, 2) for k, v in {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}.items()}


# ─── 52-week metrics ─────────────────────────────────────────────────────────

def week52_metrics(df):
    df = _flat(df)
    if df.empty or len(df) < 2 or "Close" not in df.columns:
        return {"high52": 0, "low52": 0, "current": 0, "pct_from_high": 0, "pct_from_low": 0, "position_pct": 50}
    c = df["Close"]
    n = min(252, len(c))
    high52 = float(c.rolling(n).max().iloc[-1])
    low52  = float(c.rolling(n).min().iloc[-1])
    current = float(c.iloc[-1])
    rng = high52 - low52
    return {
        "high52":        round(high52, 2),
        "low52":         round(low52, 2),
        "current":       round(current, 2),
        "pct_from_high": round((current / high52 - 1) * 100, 1) if high52 else 0,
        "pct_from_low":  round((current / low52 - 1) * 100, 1) if low52 else 0,
        "position_pct":  round((current - low52) / rng * 100, 1) if rng else 50,
    }


# ─── Volume metrics ──────────────────────────────────────────────────────────

def volume_metrics(df, avg_period=20):
    df = _flat(df)
    if df.empty or "Volume" not in df.columns or len(df) < avg_period:
        return {"vol_ratio": 1.0, "avg_volume": 0, "today_volume": 0}
    avg = float(df["Volume"].rolling(avg_period).mean().iloc[-1])
    today = float(df["Volume"].iloc[-1])
    return {"vol_ratio": round(today / avg, 2) if avg > 0 else 1.0, "avg_volume": round(avg), "today_volume": round(today)}


# ─── RRG ─────────────────────────────────────────────────────────────────────

def compute_rrg(sector_data, benchmark_df, rs_period=12, mom_period=5):
    """JdK RS-Ratio and RS-Momentum for RRG chart."""
    benchmark_df = _flat(benchmark_df)
    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        return pd.DataFrame()
    bench = benchmark_df["Close"].rename("bench")
    rows = []
    for key, df in sector_data.items():
        df = _flat(df)
        if df.empty or "Close" not in df.columns:
            continue
        if len(df) < rs_period + mom_period + 10:
            continue
        sec = df["Close"]
        aligned = pd.concat([sec.rename("sec"), bench], axis=1, join="inner").dropna()
        if len(aligned) < rs_period + mom_period:
            continue
        rs_raw    = aligned["sec"] / aligned["bench"]
        rs_smooth = rs_raw.ewm(span=rs_period, adjust=False).mean()
        rs_ref    = rs_smooth.rolling(min(len(rs_smooth), 252)).mean()
        rs_ratio  = (rs_smooth / rs_ref * 100).replace([np.inf, -np.inf], np.nan).fillna(100)
        rs_mom_r  = rs_ratio.pct_change(mom_period) * 100
        rs_mom_s  = rs_mom_r.rolling(min(len(rs_mom_r), 252)).std().replace(0, np.nan)
        rs_mom    = (rs_mom_r / rs_mom_s + 100).replace([np.inf, -np.inf], np.nan).fillna(100)
        rr_val    = float(rs_ratio.iloc[-1])
        rm_val    = float(rs_mom.iloc[-1])
        stage     = _rrg_stage(rr_val, rm_val)
        tail      = min(8, len(rs_ratio))
        rows.append({
            "sector":      key,
            "rs_ratio":    round(rr_val, 2),
            "rs_momentum": round(rm_val, 2),
            "stage":       stage,
            "trail_rr":    rs_ratio.iloc[-tail:].round(2).tolist(),
            "trail_rm":    rs_mom.iloc[-tail:].round(2).tolist(),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _rrg_stage(rs_ratio, rs_momentum):
    if rs_ratio >= 100 and rs_momentum >= 100:
        return "Leading"
    if rs_ratio >= 100 and rs_momentum < 100:
        return "Weakening"
    if rs_ratio < 100 and rs_momentum < 100:
        return "Lagging"
    return "Improving"


# ─── All indicators combined ─────────────────────────────────────────────────

def all_indicators(df):
    """Compute every indicator for one stock DataFrame."""
    df = _flat(df)
    if df.empty or "Close" not in df.columns or len(df) < 60:
        return {}
    close = df["Close"]
    ind = {}
    ind["ema9"]  = float(ema(close, 9).iloc[-1])
    ind["ema20"] = float(ema(close, 20).iloc[-1])
    ind["ema50"] = float(ema(close, 50).iloc[-1])
    ind["atr14"] = float(atr(df, 14).iloc[-1])
    adx_df = adx(df, 14)
    ind["adx"]      = float(adx_df["ADX"].iloc[-1])
    ind["di_plus"]  = float(adx_df["+DI"].iloc[-1])
    ind["di_minus"] = float(adx_df["-DI"].iloc[-1])
    ind["rsi14"] = float(rsi(close, 14).iloc[-1])
    macd_df = macd(close)
    ind["macd_hist"]   = float(macd_df["Histogram"].iloc[-1])
    ind["macd_hist_p"] = float(macd_df["Histogram"].iloc[-2]) if len(macd_df) > 1 else 0.0
    bb = bollinger_bands(close)
    ind["bb_upper"] = float(bb["Upper"].iloc[-1])
    ind["bb_lower"] = float(bb["Lower"].iloc[-1])
    ind["bb_mid"]   = float(bb["Mid"].iloc[-1])
    smi_df = smi(df)
    ind["smi"]     = float(smi_df["SMI"].iloc[-1])
    ind["smi_sig"] = float(smi_df["SMI_Signal"].iloc[-1])
    ind["close"] = float(close.iloc[-1])
    ind["open"]  = float(df["Open"].iloc[-1]) if "Open" in df.columns else 0.0
    ind["high"]  = float(df["High"].iloc[-1]) if "High" in df.columns else 0.0
    ind["low"]   = float(df["Low"].iloc[-1])  if "Low"  in df.columns else 0.0
    ind.update(week52_metrics(df))
    ind.update(volume_metrics(df))
    ind.update(pivot_levels(df))
    return ind
