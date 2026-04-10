"""
scoring.py
==========
Weighted probability scoring model.
  Signal alignment  40%
  Volume strength   20%
  RS strength       20%
  RRG stage         20%
"""

from typing import Dict, Optional

WEIGHTS = {
    "signal": 0.40,
    "volume": 0.20,
    "rs":     0.20,
    "rrg":    0.20,
}

RRG_STAGE_SCORE = {
    "Leading":   100,
    "Improving":  75,
    "Weakening":  35,
    "Lagging":    10,
    None:         50,
}

GRADE_THRESHOLDS = [
    (80, "A+"),
    (70, "A"),
    (60, "B"),
    (45, "C"),
    (0,  "D"),
]

GRADE_COLORS = {
    "A+": "#00c853",
    "A":  "#69f0ae",
    "B":  "#ffd740",
    "C":  "#ff9800",
    "D":  "#ef5350",
}


# ─────────────────────────────────────────────────────────────────────────────
# SUB-SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_signal(ind: Dict, horizon: str) -> float:
    """0-100 based on how many horizon-specific conditions are satisfied."""
    close     = ind.get("close", 0)
    ema9      = ind.get("ema9",  0)
    ema20     = ind.get("ema20", 0)
    ema50     = ind.get("ema50", 0)
    adx_v     = ind.get("adx",   0)
    rsi_v     = ind.get("rsi14", 50)
    bb_upper  = ind.get("bb_upper", 1e9)
    macd_h    = ind.get("macd_hist",   0)
    macd_hp   = ind.get("macd_hist_p", 0)
    vol_r     = ind.get("vol_ratio",   1.0)

    checks = []

    if horizon == "SWING":
        checks += [
            close > ema9,
            adx_v > 25,
            vol_r > 1.2,
            rsi_v < 68,
            close < bb_upper,
            adx_v > 20,
        ]
        if adx_v > 35:
            checks.append(True)

    elif horizon == "SHORT_TERM":
        checks += [
            close > ema20,
            close > ema50,
            adx_v >= 20,
            macd_h > macd_hp,
            macd_h > 0,
            40 < rsi_v < 75,
        ]
        if adx_v > 30:
            checks.append(True)

    elif horizon == "LONG_TERM":
        checks += [
            close > ema50,
            adx_v > 20,
            close > ema20,
            rsi_v > 35,
            vol_r >= 0.8,
            ind.get("position_pct", 50) < 85,
        ]

    if not checks:
        return 50.0
    return round(sum(checks) / len(checks) * 100, 1)


def score_volume(ind: Dict) -> float:
    r = ind.get("vol_ratio", 1.0)
    if r >= 2.5: return 100.0
    if r >= 2.0: return 90.0
    if r >= 1.5: return 75.0
    if r >= 1.2: return 60.0
    if r >= 0.8: return 45.0
    return 25.0


def score_rs(rs_val: float) -> float:
    if rs_val >= 15: return 100.0
    if rs_val >=  8: return 85.0
    if rs_val >=  3: return 70.0
    if rs_val >=  0: return 55.0
    if rs_val >= -5: return 35.0
    return 15.0


def score_rrg(stage: Optional[str]) -> float:
    return float(RRG_STAGE_SCORE.get(stage, 50))


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(ind: Dict, horizon: str,
                  rs_val: float = 0.0,
                  rrg_stage: Optional[str] = None) -> Dict:
    """
    Returns:
      score        0-100
      grade        A+/A/B/C/D
      signal_score, volume_score, rs_score, rrg_score
    """
    s_sig = score_signal(ind, horizon)
    s_vol = score_volume(ind)
    s_rs  = score_rs(rs_val)
    s_rrg = score_rrg(rrg_stage)

    composite = round(
        s_sig * WEIGHTS["signal"] +
        s_vol * WEIGHTS["volume"] +
        s_rs  * WEIGHTS["rs"]     +
        s_rrg * WEIGHTS["rrg"],
        1,
    )
    composite = max(0.0, min(100.0, composite))

    grade = "D"
    for threshold, g in GRADE_THRESHOLDS:
        if composite >= threshold:
            grade = g
            break

    return {
        "score":        composite,
        "grade":        grade,
        "signal_score": s_sig,
        "volume_score": s_vol,
        "rs_score":     s_rs,
        "rrg_score":    s_rrg,
    }
