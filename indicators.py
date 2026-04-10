"""
indicators.py
=============
All technical indicator computations from live OHLCV data.
EMA, ATR, ADX, RSI, MACD, Bollinger Bands, SMI, Pivots, RRG, Volume, 52W.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGES
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    H, L, C = df["High"], df["Low"], df["Close"]
    prev_c  = C.shift(1)
    tr = pd.concat([H - L, (H - prev_c).abs(), (L - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# ADX
# ─────────────────────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    H, L, C = df["High"], df["Low"], df["Close"]
    prev_H, prev_L, prev_C = H.shift(1), L.shift(1), C.shift(1)

    tr = pd.concat([H - L, (H - prev_C).abs(), (L - prev_C).abs()], axis=1).max(axis=1)

    dm_p = np.where((H - prev_H) > (prev_L - L), np.maximum(H - prev_H, 0), 0)
    dm_m = np.where((prev_L - L) > (H - prev_H), np.maximum(prev_L - L, 0), 0)

    dm_p = pd.Series(dm_p, index=df.index)
    dm_m = pd.Series(dm_m, index=df.index)

    atr_  = tr.ewm(span=period, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / atr_.replace(0, np.nan)
    di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / atr_.replace(0, np.nan)

    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx_v = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({"ADX": adx_v, "+DI": di_p, "-DI": di_m}, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_f  = series.ewm(span=fast,   adjust=False).mean()
    ema_s  = series.ewm(span=slow,   adjust=False).mean()
    line   = ema_f - ema_s
    sig    = line.ewm(span=signal, adjust=False).mean()
    hist   = line - sig
    return pd.DataFrame({"MACD": line, "Signal": sig, "Histogram": hist},
                        index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# BOLLINGER BANDS
# ─────────────────────────────────────────────────────────────────────────────

def bollinger_bands(series: pd.Series,
                    period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    mid   = sma(series, period)
    std   = series.rolling(period).std()
    return pd.DataFrame({
        "Upper": mid + std_dev * std,
        "Mid":   mid,
        "Lower": mid - std_dev * std,
    }, index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# SMI  (Stochastic Momentum Index)
# ─────────────────────────────────────────────────────────────────────────────

def smi(df: pd.DataFrame,
        period: int = 14, sm1: int = 3, sm2: int = 3, sig: int = 8) -> pd.DataFrame:
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    mid  = (hi + lo) / 2
    diff = df["Close"] - mid
    num  = diff.ewm(span=sm1, adjust=False).mean().ewm(span=sm2, adjust=False).mean()
    den  = (hi - lo).div(2).ewm(span=sm1, adjust=False).mean().ewm(span=sm2, adjust=False).mean()
    val  = 100 * num / den.replace(0, np.nan)
    sig_ = val.ewm(span=sig, adjust=False).mean()
    return pd.DataFrame({"SMI": val, "SMI_Signal": sig_}, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

def rs_ratio_series(stock: pd.Series, bench: pd.Series) -> pd.Series:
    """RS line rebased to 100."""
    aligned = pd.concat([stock.rename("s"), bench.rename("b")],
                        axis=1, join="inner").dropna()
    rs = aligned["s"] / aligned["b"]
    return (rs / rs.iloc[0] * 100)


def rs_score(stock: pd.Series, bench: pd.Series,
             period: int = 63) -> float:
    """% outperformance vs benchmark over `period` bars."""
    aligned = pd.concat([stock.rename("s"), bench.rename("b")],
                        axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    n = min(period, len(aligned) - 1)
    s = (aligned["s"].iloc[-1] / aligned["s"].iloc[-n] - 1) * 100
    b = (aligned["b"].iloc[-1] / aligned["b"].iloc[-n] - 1) * 100
    return round(s - b, 2)


# ─────────────────────────────────────────────────────────────────────────────
# PIVOT S/R
# ─────────────────────────────────────────────────────────────────────────────

def pivot_levels(df: pd.DataFrame) -> Dict[str, float]:
    """Classic pivot point from last complete session."""
    if df.empty or len(df) < 2:
        return {}
    row = df.iloc[-2]          # last complete candle
    H, L, C = float(row["High"]), float(row["Low"]), float(row["Close"])
    P  = (H + L + C) / 3
    R1 = 2 * P - L
    S1 = 2 * P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    return {k: round(v, 2) for k, v in {"P": P, "R1": R1, "R2": R2,
                                          "S1": S1, "S2": S2}.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 52-WEEK METRICS
# ─────────────────────────────────────────────────────────────────────────────

def week52_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or len(df) < 2:
        return {"high52": 0, "low52": 0, "current": 0,
                "pct_from_high": 0, "pct_from_low": 0, "position_pct": 50}
    c       = df["Close"]
    n       = min(252, len(c))
    high52  = float(c.rolling(n).max().iloc[-1])
    low52   = float(c.rolling(n).min().iloc[-1])
    current = float(c.iloc[-1])
    rng     = high52 - low52
    return {
        "high52":        round(high52, 2),
        "low52":         round(low52,  2),
        "current":       round(current, 2),
        "pct_from_high": round((current / high52 - 1) * 100, 1) if high52 else 0,
        "pct_from_low":  round((current / low52  - 1) * 100, 1) if low52  else 0,
        "position_pct":  round((current - low52) / rng * 100, 1) if rng else 50,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME METRICS
# ─────────────────────────────────────────────────────────────────────────────

def volume_metrics(df: pd.DataFrame, avg_period: int = 20) -> Dict:
    if df.empty or "Volume" not in df.columns or len(df) < avg_period:
        return {"vol_ratio": 1.0, "avg_volume": 0, "today_volume": 0}
    avg = float(df["Volume"].rolling(avg_period).mean().iloc[-1])
    today = float(df["Volume"].iloc[-1])
    return {
        "vol_ratio":    round(today / avg, 2) if avg > 0 else 1.0,
        "avg_volume":   round(avg),
        "today_volume": round(today),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RRG  (Relative Rotation Graph — JdK methodology)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rrg(sector_data: Dict[str, pd.DataFrame],
                benchmark_df: pd.DataFrame,
                rs_period: int = 12,
                mom_period: int = 5) -> pd.DataFrame:
    """
    Compute JdK RS-Ratio and RS-Momentum.
    RS-Ratio  = smoothed(RS) normalised around 100
    RS-Momentum = RoC of RS-Ratio normalised around 100
    """
    bench = benchmark_df["Close"].rename("bench")
    rows  = []

    for key, df in sector_data.items():
        if df.empty or len(df) < rs_period + mom_period + 10:
            continue
        sec     = df["Close"]
        aligned = pd.concat([sec.rename("sec"), bench],
                             axis=1, join="inner").dropna()
        if len(aligned) < rs_period + mom_period:
            continue

        rs_raw      = aligned["sec"] / aligned["bench"]
        rs_smooth   = rs_raw.ewm(span=rs_period, adjust=False).mean()
        rs_norm_ref = rs_smooth.rolling(min(len(rs_smooth), 252)).mean()
        rs_ratio    = (rs_smooth / rs_norm_ref * 100).replace([np.inf, -np.inf], np.nan).fillna(100)

        rs_mom_raw  = rs_ratio.pct_change(mom_period) * 100
        rs_mom_std  = rs_mom_raw.rolling(min(len(rs_mom_raw), 252)).std().replace(0, np.nan)
        rs_mom      = (rs_mom_raw / rs_mom_std + 100).replace([np.inf, -np.inf], np.nan).fillna(100)

        rr_val  = float(rs_ratio.iloc[-1])
        rm_val  = float(rs_mom.iloc[-1])
        stage   = _rrg_stage(rr_val, rm_val)

        # Trail (last 8 points)
        tail = min(8, len(rs_ratio))
        rows.append({
            "sector":      key,
            "rs_ratio":    round(rr_val, 2),
            "rs_momentum": round(rm_val, 2),
            "stage":       stage,
            "trail_rr":    rs_ratio.iloc[-tail:].round(2).tolist(),
            "trail_rm":    rs_mom.iloc[-tail:].round(2).tolist(),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _rrg_stage(rs_ratio: float, rs_momentum: float) -> str:
    if rs_ratio >= 100 and rs_momentum >= 100:  return "Leading"
    if rs_ratio >= 100 and rs_momentum <  100:  return "Weakening"
    if rs_ratio <  100 and rs_momentum <  100:  return "Lagging"
    return "Improving"


RRG_COLORS = {"Leading": "#00c853", "Weakening": "#ff9800",
              "Lagging": "#ef5350", "Improving": "#2196f3"}

# Alias so app.py can import either name
STAGE_COLORS = RRG_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# ALL INDICATORS COMBINED  (for trade engine)
# ─────────────────────────────────────────────────────────────────────────────

def all_indicators(df: pd.DataFrame) -> Dict:
    """Compute every indicator needed for scoring + setup generation."""
    if df.empty or len(df) < 60:
        return {}

    close = df["Close"]
    ind: Dict = {}

    # EMAs
    ind["ema9"]  = float(ema(close,  9).iloc[-1])
    ind["ema20"] = float(ema(close, 20).iloc[-1])
    ind["ema50"] = float(ema(close, 50).iloc[-1])

    # ATR
    ind["atr14"] = float(atr(df, 14).iloc[-1])

    # ADX
    adx_df        = adx(df, 14)
    ind["adx"]    = float(adx_df["ADX"].iloc[-1])
    ind["di_plus"]  = float(adx_df["+DI"].iloc[-1])
    ind["di_minus"] = float(adx_df["-DI"].iloc[-1])

    # RSI
    ind["rsi14"] = float(rsi(close, 14).iloc[-1])

    # MACD
    macd_df            = macd(close)
    ind["macd_hist"]   = float(macd_df["Histogram"].iloc[-1])
    ind["macd_hist_p"] = float(macd_df["Histogram"].iloc[-2]) if len(macd_df) > 1 else 0.0

    # Bollinger
    bb              = bollinger_bands(close)
    ind["bb_upper"] = float(bb["Upper"].iloc[-1])
    ind["bb_lower"] = float(bb["Lower"].iloc[-1])
    ind["bb_mid"]   = float(bb["Mid"].iloc[-1])

    # SMI
    smi_df          = smi(df)
    ind["smi"]      = float(smi_df["SMI"].iloc[-1])
    ind["smi_sig"]  = float(smi_df["SMI_Signal"].iloc[-1])

    # Price
    ind["close"] = float(close.iloc[-1])
    ind["open"]  = float(df["Open"].iloc[-1])
    ind["high"]  = float(df["High"].iloc[-1])
    ind["low"]   = float(df["Low"].iloc[-1])

    # 52-week
    ind.update(week52_metrics(df))

    # Volume
    ind.update(volume_metrics(df))

    # Pivots
    ind.update(pivot_levels(df))

    return ind
