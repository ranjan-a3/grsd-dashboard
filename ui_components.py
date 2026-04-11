"""
ui_components.py
================
All Streamlit UI components: cards, charts, RRG, heatmap, badges, S/R chart.
Bloomberg dark terminal aesthetic.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from indicators import RRG_COLORS, ema, bollinger_bands, atr
from scoring    import GRADE_COLORS

def _hex_rgba(hex_color, alpha=0.4):
    """Convert #rrggbb to rgba(r,g,b,alpha) for plotly compatibility."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(128,128,128,{alpha})"




# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

T = {
    "bg0":    "#0d1117",
    "bg1":    "#161b22",
    "bg2":    "#21262d",
    "border": "#30363d",
    "muted":  "#8b949e",
    "bright": "#e6edf3",
    "green":  "#00c853",
    "red":    "#f44336",
    "orange": "#ff9800",
    "blue":   "#2196f3",
    "gold":   "#ffd700",
    "accent": "#38bdf8",
}

STAGE_COLORS = RRG_COLORS
STAGE_EMOJI  = {
    "Leading":   "🟢",
    "Weakening": "🟠",
    "Lagging":   "🔴",
    "Improving": "🔵",
}


# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, .stApp { background-color: #0d1117; color: #e6edf3; }

    .stTabs [data-baseweb="tab-list"] {
        background: #161b22; border-radius: 8px; padding: 4px; gap: 4px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif; font-weight: 500;
        font-size: 0.82rem; color: #8b949e; border-radius: 6px; padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #0d1117 !important; color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    [data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    [data-testid="stSidebar"] * { color: #e6edf3; }
    .stButton > button {
        background: #161b22; color: #e6edf3; border: 1px solid #30363d;
        border-radius: 6px; font-family: 'DM Sans', sans-serif; transition: all 0.2s;
    }
    .stButton > button:hover { border-color: #38bdf8; color: #38bdf8; }

    .card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 18px; margin-bottom: 14px; font-family: 'DM Sans', sans-serif;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #3d444d; }

    .ticker { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e6edf3; }

    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.68rem; font-weight: 700; font-family: 'DM Sans', sans-serif;
        text-transform: uppercase; letter-spacing: 0.04em;
    }

    .level-box {
        background: #0d1117; border-radius: 8px; padding: 10px 8px; text-align: center;
    }
    .level-label { font-size: 0.65rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
    .level-value { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; }

    .metric-mini {
        background: #0d1117; border-radius: 6px; padding: 7px; text-align: center;
    }
    .metric-mini .k { font-size: 0.6rem; color: #8b949e; }
    .metric-mini .v { font-family: 'JetBrains Mono', monospace; font-size: 0.84rem; font-weight: 700; }

    .prob-bar-bg { background: #0d1117; border-radius: 4px; height: 7px; overflow: hidden; margin-top: 4px; }
    .prob-bar-fill { height: 100%; border-radius: 4px; }

    .w52-track { background: #30363d; border-radius: 4px; height: 9px; position: relative; margin: 4px 0; }
    .w52-fill { position: absolute; height: 100%; border-radius: 4px; }
    .w52-dot { position: absolute; width: 13px; height: 13px; border-radius: 50%; top: -2px; transform: translateX(-50%); border: 2px solid #0d1117; }

    .section-hdr {
        font-size: 0.65rem; color: #8b949e; text-transform: uppercase;
        letter-spacing: 0.12em; border-bottom: 1px solid #30363d;
        padding-bottom: 8px; margin-bottom: 14px; font-weight: 700;
    }

    .ntz-warn {
        background: #1f1500; border: 1px solid #ff9800; border-radius: 8px;
        padding: 10px 14px; font-size: 0.8rem; color: #ff9800; margin: 8px 0;
    }

    .sync-banner {
        background: rgba(33,150,243,0.08); border: 1px solid rgba(33,150,243,0.3);
        border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
    }

    div[data-testid="stHorizontalBlock"] { gap: 10px; }

    .stDataFrame { background: #161b22; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rs_color(val: float) -> str:
    if val > 5:   return T["green"]
    if val > 0:   return "#69f0ae"
    if val > -5:  return T["orange"]
    return T["red"]

def _vol_color(ratio: float) -> str:
    if ratio >= 2.0:  return T["green"]
    if ratio >= 1.2:  return "#ffd740"
    return T["red"]

def _rsi_color(rsi: float) -> str:
    if rsi > 68: return T["red"]
    if rsi > 60: return "#ffd740"
    return T["green"]

def stage_badge_html(stage: str) -> str:
    color = STAGE_COLORS.get(stage, T["muted"])
    emoji = STAGE_EMOJI.get(stage, "⚪")
    return (f'<span style="color:{color};font-weight:700;font-size:0.78rem;">'
            f'{emoji} {stage}</span>')

def horizon_badge_html(horizon: str) -> str:
    cfg = {
        "SWING":      ("⚡", "Swing",      "#1a2744", "#64b5f6"),
        "SHORT_TERM": ("📈", "Short Term", "#1a3326", "#69f0ae"),
        "LONG_TERM":  ("🏦", "Long Term",  "#2d2011", "#ffd700"),
    }
    ic, lb, bg, clr = cfg.get(horizon, ("", horizon, "#161b22", "#888"))
    return (f'<span style="font-size:0.68rem;font-weight:700;padding:2px 9px;'
            f'border-radius:20px;background:{bg};color:{clr};'
            f'border:1px solid {clr}44;font-family:monospace;">'
            f'{ic} {lb}</span>')

def prob_bar_html(score: float, grade: str) -> str:
    clr = GRADE_COLORS.get(grade, T["muted"])
    return f"""
<div style="margin:6px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <span style="font-size:0.65rem;color:{T['muted']};text-transform:uppercase;letter-spacing:0.08em;">Probability</span>
    <span style="font-family:monospace;font-weight:700;color:{clr};font-size:0.88rem;">
      {score:.0f}%
      <span style="background:{clr}22;padding:1px 7px;border-radius:4px;border:1px solid {clr}44;font-size:0.7rem;">{grade}</span>
    </span>
  </div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{score}%;background:linear-gradient(90deg,{clr}88,{clr});"></div>
  </div>
</div>"""

def w52_bar_html(position_pct: float, pct_from_high: float) -> str:
    if position_pct <= 20:
        clr, zone = T["green"], "Value Zone"
    elif position_pct >= 85:
        clr, zone = T["gold"], "Breakout Zone"
    else:
        clr, zone = T["accent"], ""
    zone_html = f'<span style="color:{clr};font-size:0.65rem;">{zone}</span>' if zone else ""
    return f"""
<div style="margin:8px 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:{T['muted']};margin-bottom:3px;">
    <span>52W Low</span>{zone_html}<span>52W High</span>
  </div>
  <div class="w52-track">
    <div class="w52-fill" style="width:{position_pct}%;background:linear-gradient(90deg,#1a3a6b,{clr});"></div>
    <div class="w52-dot" style="left:{position_pct}%;background:{clr};"></div>
  </div>
  <div style="text-align:right;font-size:0.63rem;color:{T['muted']};margin-top:2px;">{pct_from_high:+.1f}% from 52W High</div>
</div>"""

def volume_html(vol_ratio: float, delivery: float) -> str:
    vc = _vol_color(vol_ratio)
    dc = T["green"] if delivery >= 60 else ("#ffd740" if delivery >= 40 else T["red"])
    return (f'<span style="font-family:monospace;font-size:0.76rem;color:{vc};">Vol {vol_ratio:.1f}×</span>'
            f'&nbsp;&nbsp;|&nbsp;&nbsp;'
            f'<span style="font-family:monospace;font-size:0.76rem;color:{dc};">Del {delivery:.0f}%</span>')


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SETUP CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_trade_card(s: Dict):
    currency = "₹" if s.get("market") == "INDIA" else "$"
    sc = T["green"] if "Strong" in s["status"] else (T["orange"] if "Watch" in s["status"] else T["red"])

    # No-trade zone
    ntz = s["adx"] < 20
    ntz_html = ('<div class="ntz-warn">⚠️ No-Trade Zone: ADX &lt; 20 (low trend strength)</div>'
                if ntz else "")

    # Header card
    st.markdown(f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div>
      <span class="ticker">{s['ticker']}</span>&nbsp;&nbsp;
      {horizon_badge_html(s['horizon'])}&nbsp;&nbsp;
      <span style="font-size:0.72rem;color:{T['muted']};">{s['sector_label']}</span>
    </div>
    <div style="text-align:right;">
      <span style="color:{sc};font-size:0.8rem;font-weight:700;">{s['status']}</span><br>
      {stage_badge_html(s['rrg_stage'])}
    </div>
  </div>
  {ntz_html}
</div>""", unsafe_allow_html=True)

    # 3-Box Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="level-box" style="border:1px solid #1e3a5f;">
          <div class="level-label">Entry</div>
          <div class="level-value" style="color:#64b5f6;">{currency}{s['entry']:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="level-box" style="border:1px solid #4a1515;">
          <div class="level-label">Stop Loss</div>
          <div class="level-value" style="color:{T['red']};">{currency}{s['sl']:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="level-box" style="border:1px solid #1b4332;">
          <div class="level-label">Target</div>
          <div class="level-value" style="color:{T['green']};">{currency}{s['target']:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    st.markdown(prob_bar_html(s["prob_score"], s["grade"]), unsafe_allow_html=True)

    # Metrics row
    rr_c  = T["green"] if s["rr"] >= 2 else ("#ffd740" if s["rr"] >= 1.5 else T["red"])
    adx_c = T["green"] if s["adx"] > 25 else ("#ffd740" if s["adx"] > 20 else T["red"])

    st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:8px 0;">
  <div class="metric-mini"><div class="k">R:R</div><div class="v" style="color:{rr_c};">{s['rr']:.1f}:1</div></div>
  <div class="metric-mini"><div class="k">ADX</div><div class="v" style="color:{adx_c};">{s['adx']:.0f}</div></div>
  <div class="metric-mini"><div class="k">RSI</div><div class="v" style="color:{_rsi_color(s['rsi14'])};">{s['rsi14']:.0f}</div></div>
  <div class="metric-mini"><div class="k">RS%</div><div class="v" style="color:{_rs_color(s['rs_value'])};">{s['rs_value']:+.1f}</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown(volume_html(s["vol_ratio"], s["delivery"]), unsafe_allow_html=True)
    st.markdown(w52_bar_html(s["position_pct"], s["pct_from_high"]), unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#30363d;margin:4px 0 0 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RRG CHART
# ─────────────────────────────────────────────────────────────────────────────

def render_rrg_chart(rrg_df: pd.DataFrame, sector_labels: Dict[str, str], title: str = "Relative Rotation Graph"):
    if rrg_df.empty:
        st.info("Not enough data for RRG. Fetching...")
        return

    fig = go.Figure()

    # Quadrant fills
    for (x0, x1, y0, y1, rgba, lbl, lx, ly) in [
        (100, 115, 100, 115, "rgba(0,200,83,0.06)",   "LEADING",   112, 113),
        (100, 115,  85, 100, "rgba(255,152,0,0.06)",  "WEAKENING", 112,  87),
        ( 85, 100,  85, 100, "rgba(244,67,54,0.06)",  "LAGGING",    88,  87),
        ( 85, 100, 100, 115, "rgba(33,150,243,0.06)", "IMPROVING",  88, 113),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=rgba, line=dict(width=0), layer="below")
        fig.add_annotation(x=lx, y=ly, text=lbl, showarrow=False,
                           font=dict(size=9, color="rgba(255,255,255,0.2)",
                                     family="JetBrains Mono"))

    fig.add_hline(y=100, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))
    fig.add_vline(x=100, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))

    for _, row in rrg_df.iterrows():
        color  = STAGE_COLORS.get(row["stage"], T["muted"])
        label  = sector_labels.get(row["sector"], row["sector"])
        sz     = 14 + abs(row["rs_ratio"] - 100) * 0.6

        # Trail
        trail_x = row.get("trail_rr", [row["rs_ratio"]])
        trail_y = row.get("trail_rm", [row["rs_momentum"]])
        if len(trail_x) > 1:
            fig.add_trace(go.Scatter(
                x=trail_x, y=trail_y, mode="lines",
                line=dict(color=_hex_rgba(color, 0.4), width=1.5),
                showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=[row["rs_ratio"]], y=[row["rs_momentum"]],
            mode="markers+text",
            marker=dict(size=sz, color=color,
                        line=dict(color="rgba(0,0,0,0.5)", width=1.5)),
            text=[label[:6]], textposition="top center",
            textfont=dict(size=8.5, color=T["bright"], family="JetBrains Mono"),
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>Stage: {row['stage']}<br>"
                f"RS-Ratio: {row['rs_ratio']}<br>"
                f"RS-Mom: {row['rs_momentum']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=T["bright"]), x=0),
        plot_bgcolor=T["bg0"], paper_bgcolor=T["bg1"],
        font=dict(color=T["bright"], family="DM Sans"),
        xaxis=dict(title="RS-Ratio →", range=[85, 115], gridcolor=T["border"],
                   tickfont=dict(size=9, family="JetBrains Mono")),
        yaxis=dict(title="RS-Momentum →", range=[85, 115], gridcolor=T["border"],
                   tickfont=dict(size=9, family="JetBrains Mono")),
        height=520, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE LINE CHART
# ─────────────────────────────────────────────────────────────────────────────

def render_performance_chart(
    sector_data:  Dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    labels:       Dict[str, str],
    title:        str = "Sector Relative Performance vs Benchmark",
):
    if not sector_data or benchmark_df.empty:
        st.info("Fetching performance data...")
        return

    bench = benchmark_df["Close"].rename("bench")
    fig   = go.Figure()

    palette = ["#00c853","#2196f3","#ffd740","#ff9800","#f44336",
               "#e91e63","#9c27b0","#00bcd4","#8bc34a","#ff5722","#607d8b"]

    for i, (key, df) in enumerate(sector_data.items()):
        if df.empty:
            continue
        aligned = pd.concat([df["Close"], bench], axis=1, join="inner").dropna()
        if len(aligned) < 5:
            continue
        rel = (aligned["Close"] / aligned["Close"].iloc[0] - 1) * 100 \
            - (aligned["bench"]  / aligned["bench"].iloc[0]  - 1) * 100
        current = rel.iloc[-1]
        color   = palette[i % len(palette)] if current >= 0 else T["red"]
        label   = labels.get(key, key)

        fig.add_trace(go.Scatter(
            x=aligned.index, y=rel.round(2), mode="lines",
            name=f"{label} ({current:+.1f}%)",
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=T["bright"]), x=0),
        plot_bgcolor=T["bg0"], paper_bgcolor=T["bg1"],
        font=dict(color=T["bright"], family="DM Sans"),
        xaxis=dict(gridcolor=T["border"], tickfont=dict(size=9)),
        yaxis=dict(title="Relative Return (%)", gridcolor=T["border"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8.5), orientation="v", x=1.01),
        height=440, hovermode="x unified",
        margin=dict(l=60, r=160, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def render_heatmap(heatmap_df: pd.DataFrame, title: str = "Momentum Heatmap"):
    if heatmap_df.empty:
        st.info("Loading heatmap...")
        return

    pivot = heatmap_df.pivot_table(
        values="score", index="sector", columns="ticker", aggfunc="first"
    )
    if pivot.empty:
        st.info("Not enough stocks loaded yet.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#4a0000"], [0.25, "#b71c1c"], [0.45, "#f44336"],
            [0.5,  "#37474f"],
            [0.55, "#1b5e20"], [0.75, "#2e7d32"], [1.0,  "#00c853"],
        ],
        zmid=50,
        text=[[f"{v:.0f}" if not np.isnan(v) else "-" for v in r] for r in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=9, family="JetBrains Mono"),
        hovertemplate="<b>%{y} | %{x}</b><br>Score: %{z:.0f}<extra></extra>",
        colorbar=dict(title="Score", tickfont=dict(size=9, color=T["bright"]),
                      titlefont=dict(size=10, color=T["bright"])),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=T["bright"]), x=0),
        plot_bgcolor=T["bg0"], paper_bgcolor=T["bg1"],
        font=dict(color=T["bright"], family="DM Sans"),
        xaxis=dict(tickfont=dict(size=8, family="JetBrains Mono"), tickangle=-45),
        yaxis=dict(tickfont=dict(size=10, family="DM Sans")),
        height=max(350, len(pivot.index) * 48 + 100),
        margin=dict(l=90, r=80, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK + INDICATORS CHART
# ─────────────────────────────────────────────────────────────────────────────

def render_candlestick(df: pd.DataFrame, ticker: str, ind: Dict, lookback: int = 90):
    if df.empty:
        st.info("No data for this ticker.")
        return

    plot_df = df.tail(lookback).copy()
    close   = df["Close"]

    # Subplot: price + volume
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # Candles
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"],   close=plot_df["Close"],
        name=ticker,
        increasing_line_color=T["green"], decreasing_line_color=T["red"],
        increasing_fillcolor=T["green"],  decreasing_fillcolor=T["red"],
    ), row=1, col=1)

    # EMAs
    for period, color, lw in [(9, "#64b5f6", 1.5), (20, "#ffd740", 1.5), (50, "#ff9800", 1.8)]:
        e_vals = ema(close, period).tail(lookback)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=e_vals, mode="lines",
            name=f"EMA{period}", line=dict(color=color, width=lw),
        ), row=1, col=1)

    # Bollinger
    bb = bollinger_bands(close)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=bb["Upper"].tail(lookback), mode="lines",
        name="BB Upper", line=dict(color="rgba(120,120,255,0.4)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=bb["Lower"].tail(lookback), mode="lines",
        name="BB Lower", line=dict(color="rgba(120,120,255,0.4)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(50,50,130,0.06)",
    ), row=1, col=1)

    # S/R levels
    for level_name, val, lc in [
        ("R2", ind.get("R2", 0), "rgba(244,67,54,0.9)"),
        ("R1", ind.get("R1", 0), "rgba(255,152,0,0.9)"),
        ("S1", ind.get("S1", 0), "rgba(33,150,243,0.9)"),
        ("S2", ind.get("S2", 0), "rgba(0,200,83,0.9)"),
    ]:
        if val and val > 0:
            fig.add_hline(y=val, row=1, col=1,
                          line=dict(color=lc, width=1, dash="dash"),
                          annotation_text=f" {level_name}: {val:.0f}",
                          annotation_position="right",
                          annotation_font=dict(size=9, color=lc))

    # Volume bars
    vol_colors = [T["green"] if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i]
                  else T["red"] for i in range(len(plot_df))]
    if "Volume" in plot_df.columns:
        fig.add_trace(go.Bar(
            x=plot_df.index, y=plot_df["Volume"],
            marker_color=vol_colors, name="Volume", opacity=0.6,
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Technical Chart", font=dict(size=13, color=T["bright"]), x=0),
        plot_bgcolor=T["bg0"], paper_bgcolor=T["bg1"],
        font=dict(color=T["bright"], family="DM Sans"),
        xaxis=dict(gridcolor=T["border"], rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor=T["border"]),
        yaxis=dict(gridcolor=T["border"], tickformat=".0f"),
        yaxis2=dict(gridcolor=T["border"], title="Volume"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=8.5), orientation="h", y=-0.15),
        height=560, hovermode="x unified",
        margin=dict(l=60, r=80, t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# RS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def render_rs_table(rs_df: pd.DataFrame, labels: Dict[str, str]):
    if rs_df.empty:
        st.info("Computing RS scores...")
        return

    display = rs_df.copy()
    display.index = [labels.get(k, k) for k in display.index]

    def style_cell(val):
        if pd.isna(val): return "color:#8b949e;"
        if val > 5:  return "color:#00c853;font-weight:700;"
        if val > 0:  return "color:#69f0ae;"
        if val > -5: return "color:#ff9800;"
        return "color:#f44336;font-weight:700;"

    styled = display.style.map(style_cell).format("{:+.1f}%")
    st.dataframe(styled, use_container_width=True, height=380)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR SECTOR STAGES
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_sectors(rrg_df: pd.DataFrame, labels: Dict[str, str]):
    st.sidebar.markdown("### 📡 Sector Stages")
    if rrg_df.empty:
        st.sidebar.info("Loading RRG data...")
        return
    for _, row in rrg_df.iterrows():
        label = labels.get(row["sector"], row["sector"])
        stage = row["stage"]
        color = STAGE_COLORS.get(stage, T["muted"])
        emoji = STAGE_EMOJI.get(stage, "⚪")
        st.sidebar.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:4px 0;border-bottom:1px solid #30363d;">'
            f'<span style="font-size:0.78rem;">{label}</span>'
            f'<span style="color:{color};font-size:0.74rem;font-weight:700;">{emoji} {stage}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THEME CARD
# ─────────────────────────────────────────────────────────────────────────────

def render_theme_card(theme: str, us_etf: str, india_sec: str,
                      us_stage: str, india_stage: str, signal: str):
    both_lead = (us_stage == "Leading" and india_stage == "Leading")
    both_bull  = (us_stage in ("Leading","Improving") and india_stage in ("Leading","Improving"))
    border = T["green"] if both_lead else (T["blue"] if both_bull else T["border"])
    bg     = "rgba(0,200,83,0.07)" if both_lead else ("rgba(33,150,243,0.07)" if both_bull else T["bg1"])

    st.markdown(f"""
<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:15px;margin-bottom:10px;">
  <div style="font-family:'DM Sans',sans-serif;font-weight:700;font-size:0.9rem;color:{T['bright']};margin-bottom:8px;">
    {'🌍 ' if both_lead else '⚠️ ' if not both_bull else '📈 '}{theme}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.78rem;">
    <div>
      <span style="color:{T['muted']};">US ({us_etf})</span><br>
      {stage_badge_html(us_stage)}
    </div>
    <div>
      <span style="color:{T['muted']};">India ({india_sec})</span><br>
      {stage_badge_html(india_stage)}
    </div>
  </div>
  <div style="margin-top:8px;font-size:0.72rem;color:{T['muted']};">{signal}</div>
</div>""", unsafe_allow_html=True)
