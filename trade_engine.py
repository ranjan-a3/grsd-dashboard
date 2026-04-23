"""
trade_engine.py
===============
Live trade setup engine.
Scans all sectors/stocks with real OHLCV, applies horizon logic,
computes Entry / SL / Target and ranks by probability score.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

def _flat(df):
    import pandas as pd
    if df is None or (hasattr(df,"empty") and df.empty):
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df



from indicators import all_indicators, rs_score
from scoring   import compute_score
from data_loader import (
    INDIA_SECTORS, US_SECTORS,
    sector_label, estimate_delivery_pct,
)


# ─────────────────────────────────────────────────────────────────────────────
# DISQUALIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def is_disqualified(ind: Dict, horizon: str) -> Tuple[bool, str]:
    close   = ind.get("close", 0)
    ema20   = ind.get("ema20", 0)
    ema50   = ind.get("ema50", 0)
    adx_v   = ind.get("adx",   0)
    rsi_v   = ind.get("rsi14", 50)
    bb_up   = ind.get("bb_upper", 1e9)
    macd_h  = ind.get("macd_hist",   0)
    macd_hp = ind.get("macd_hist_p", 0)

    if horizon == "SWING":
        if rsi_v > 68:         return True, "RSI overbought (>68)"
        if close > bb_up:      return True, "Price above BB Upper"
        if adx_v < 15:         return True, "ADX too weak (<15)"

    elif horizon == "SHORT_TERM":
        if adx_v < 20:         return True, "ADX < 20 — no trend"
        if close < ema50:      return True, "Price below EMA50"
        if macd_h < macd_hp and macd_h < 0:
                               return True, "MACD falling and negative"

    elif horizon == "LONG_TERM":
        if close < ema50:      return True, "Price below EMA50"
        if adx_v < 15:         return True, "ADX too weak"

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY / SL / TARGET
# ─────────────────────────────────────────────────────────────────────────────

def compute_levels(ind: Dict, horizon: str) -> Dict:
    close   = ind.get("close",  0)
    ema20   = ind.get("ema20",  close)
    ema50   = ind.get("ema50",  close)
    atr14   = ind.get("atr14",  close * 0.02)
    high52  = ind.get("high52", close * 1.5)
    pct_fh  = ind.get("pct_from_high", -20)

    if horizon == "SWING":
        entry  = round(close, 2)
        sl     = round(close - 2 * atr14, 2)
        risk   = close - sl
        target = round(close + max(1.5 * atr14, 2 * risk), 2)

    elif horizon == "SHORT_TERM":
        entry  = round(close, 2)
        sl     = round(ema20 * 0.99, 2)
        risk   = close - sl
        if pct_fh >= -15 and high52 > close:
            target = round(high52 * 0.99, 2)
        else:
            target = round(close + 2.5 * risk, 2)

    elif horizon == "LONG_TERM":
        entry  = round(close, 2)
        sl     = round(ema50 * 0.97, 2)
        risk   = close - sl
        target = round(close * 1.35, 2)

    else:
        entry, sl, target = close, close * 0.95, close * 1.10

    risk   = max(entry - sl, 0.01)
    reward = target - entry
    rr     = round(reward / risk, 2) if risk > 0 else 0

    return {"entry": entry, "sl": sl, "target": target,
            "risk": round(risk, 2), "reward": round(reward, 2), "rr": rr}


# ─────────────────────────────────────────────────────────────────────────────
# QUALIFICATION STATUS
# ─────────────────────────────────────────────────────────────────────────────

def qualification_status(ind: Dict, horizon: str,
                          rrg_stage: Optional[str]) -> Tuple[bool, str]:
    disq, reason = is_disqualified(ind, horizon)
    if disq:
        return False, f"Disqualified: {reason}"

    close   = ind.get("close", 0)
    ema9    = ind.get("ema9",  0)
    ema20   = ind.get("ema20", 0)
    ema50   = ind.get("ema50", 0)
    adx_v   = ind.get("adx",  0)
    vol_r   = ind.get("vol_ratio", 1.0)
    macd_h  = ind.get("macd_hist",   0)
    macd_hp = ind.get("macd_hist_p", 0)

    pts = 0
    if horizon == "SWING":
        if close > ema9:  pts += 1
        if adx_v > 25:    pts += 1
        if vol_r > 1.5:   pts += 1
        threshold = 2

    elif horizon == "SHORT_TERM":
        if close > ema20:                           pts += 1
        if rrg_stage == "Leading":                  pts += 1
        if macd_h > macd_hp:                        pts += 1
        if adx_v >= 25:                             pts += 1
        threshold = 3

    elif horizon == "LONG_TERM":
        if close > ema50:                           pts += 1
        if rrg_stage in ("Leading", "Improving"):   pts += 1
        if vol_r >= 1.0:                            pts += 1
        threshold = 2

    else:
        threshold = 2

    if pts >= threshold:
        return True, "Strong Buy"
    elif pts > 0:
        return True, "Watch"
    return False, "Weak"


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE STOCK SETUP
# ─────────────────────────────────────────────────────────────────────────────

def build_setup(
    ticker:     str,
    df:         pd.DataFrame,
    bench_cls:  pd.Series,
    sector_key: str,
    rrg_stage:  Optional[str],
    horizon:    str,
    market:     str = "INDIA",
) -> Optional[Dict]:
    if df.empty or len(df) < 60:
        return None

    ind = all_indicators(df)
    if not ind:
        return None

    df_flat = _flat(df)
    if df_flat is None or df_flat.empty or "Close" not in df_flat.columns:
        return None
    rs_val = rs_score(df_flat["Close"], bench_cls, period=63)
    ind["rs_value"] = rs_val

    qualified, status = qualification_status(ind, horizon, rrg_stage)
    levels  = compute_levels(ind, horizon)
    prob    = compute_score(ind, horizon, rs_val, rrg_stage)
    deliv   = estimate_delivery_pct(df)
    slabel  = sector_label(sector_key, market)

    currency = "INR" if market == "INDIA" else "USD"

    return {
        "ticker":       ticker,
        "sector":       sector_key,
        "sector_label": slabel,
        "horizon":      horizon,
        "market":       market,
        "currency":     currency,
        "qualified":    qualified,
        "status":       status,
        "rrg_stage":    rrg_stage or "Unknown",

        # Levels
        "entry":  levels["entry"],
        "sl":     levels["sl"],
        "target": levels["target"],
        "rr":     levels["rr"],
        "risk":   levels["risk"],
        "reward": levels["reward"],

        # Scores
        "prob_score":   prob["score"],
        "grade":        prob["grade"],
        "signal_score": prob["signal_score"],
        "volume_score": prob["volume_score"],
        "rs_score_val": prob["rs_score"],
        "rrg_score":    prob["rrg_score"],

        # Indicators
        "close":     ind.get("close",  0),
        "ema9":      ind.get("ema9",   0),
        "ema20":     ind.get("ema20",  0),
        "ema50":     ind.get("ema50",  0),
        "adx":       round(ind.get("adx",   0), 1),
        "rsi14":     round(ind.get("rsi14", 0), 1),
        "vol_ratio": ind.get("vol_ratio", 1.0),
        "delivery":  deliv,
        "rs_value":  rs_val,
        "atr14":     round(ind.get("atr14", 0), 2),
        "bb_upper":  round(ind.get("bb_upper", 0), 2),
        "bb_lower":  round(ind.get("bb_lower", 0), 2),
        "macd_hist": round(ind.get("macd_hist", 0), 4),
        "smi":       round(ind.get("smi", 0), 1),

        # 52-week
        "high52":       ind.get("high52",       0),
        "low52":        ind.get("low52",        0),
        "position_pct": ind.get("position_pct", 50),
        "pct_from_high":ind.get("pct_from_high", 0),

        # Pivots
        "S1": ind.get("S1", 0),
        "S2": ind.get("S2", 0),
        "R1": ind.get("R1", 0),
        "R2": ind.get("R2", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL UNIVERSE SCAN
# ─────────────────────────────────────────────────────────────────────────────

def scan_universe(
    universe:     Dict[str, Dict[str, pd.DataFrame]],
    benchmark_df: pd.DataFrame,
    rrg_df:       pd.DataFrame,
    horizon:      str,
    market:       str = "INDIA",
    min_score:    float = 35.0,
) -> pd.DataFrame:
    """
    Scan all sectors and stocks. Returns ranked DataFrame of live setups.
    """
    benchmark_df = _flat(benchmark_df)
    bench_cls = benchmark_df["Close"] if "Close" in benchmark_df.columns else benchmark_df.iloc[:,0]

    rrg_lookup: Dict[str, str] = {}
    if not rrg_df.empty and "sector" in rrg_df.columns:
        for _, row in rrg_df.iterrows():
            rrg_lookup[row["sector"]] = row["stage"]

    setups: List[Dict] = []
    for sector_key, stocks in universe.items():
        rrg_stage = rrg_lookup.get(sector_key)
        for ticker, df in stocks.items():
            try:
                setup = build_setup(
                    ticker, df, bench_cls,
                    sector_key, rrg_stage, horizon, market,
                )
                if setup and setup["qualified"] and setup["prob_score"] >= min_score:
                    setups.append(setup)
            except Exception:
                pass

    if not setups:
        return pd.DataFrame()

    df_out = pd.DataFrame(setups)
    df_out.sort_values("prob_score", ascending=False, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM SCORE (for heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def momentum_score(ind: Dict) -> float:
    score = 0.0
    score += min(ind.get("rsi14", 50), 80) / 80 * 25
    score += min(ind.get("adx",   20), 50) / 50 * 20
    score += min(ind.get("vol_ratio", 1.0), 3.0) / 3.0 * 20
    score += (1 if ind.get("close", 0) > ind.get("ema9",  0) else 0) * 15
    score += (1 if ind.get("close", 0) > ind.get("ema20", 0) else 0) * 10
    score += (1 if ind.get("macd_hist", 0) > 0 else 0) * 10
    return round(score, 1)
