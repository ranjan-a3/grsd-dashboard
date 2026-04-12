"""
app.py
======
Global Relative Strength & Momentum Dashboard
Main Streamlit entry point.

Run:
    streamlit run app.py

Requirements:
    pip install streamlit yfinance pandas numpy plotly requests
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
from typing import Dict, Optional

# ── Local modules ─────────────────────────────────────────────────────────────
from data_loader import (
    INDIA_SECTORS, US_SECTORS, INDIA_BENCHMARK, US_BENCHMARK,
    fetch_single, fetch_sector_indices, fetch_sector_stocks,
    fetch_full_universe, compute_relative_perf, all_sector_labels,
)
from indicators import (
    compute_rrg, all_indicators, rs_score,
    week52_metrics, volume_metrics, rsi, macd, smi, ema,
    STAGE_COLORS as RRG_STAGE_COLORS,
)
from trade_engine import scan_universe, momentum_score, build_setup
from ui_components import (
    inject_css, render_trade_card, render_rrg_chart,
    render_performance_chart, render_heatmap, render_candlestick,
    render_rs_table, render_sidebar_sectors, render_theme_card,
    stage_badge_html, STAGE_EMOJI, T,
)
from scoring import GRADE_COLORS

def _safe_close(df):
    """Safely get Close column regardless of yfinance column format."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" in df.columns:
        return df["Close"]
    return pd.Series(dtype=float)

def _safe_flat(df):
    """Flatten MultiLevel columns."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    return df



# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GRS Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TIMESTAMP
# ─────────────────────────────────────────────────────────────────────────────

def live_timestamp(country: str = "INDIA") -> str:
    lbl = {"INDIA": "IST", "US": "EST"}.get(country, "IST")
    try:
        if HAS_PYTZ:
            tz_map = {"INDIA": "Asia/Kolkata", "US": "America/New_York"}
            tz  = pytz.timezone(tz_map.get(country, "Asia/Kolkata"))
            now = datetime.now(tz)
        else:
            now = datetime.utcnow()
        return now.strftime(f"%A, %d %B %Y - %H:%M {lbl}")
    except Exception:
        return datetime.utcnow().strftime("%d %B %Y %H:%M UTC")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

# Choose active market for timestamp
_active_country = st.session_state.get("active_country", "INDIA")

# Slim date banner
st.markdown(f"""
<div style="background:{T['bg1']};border-bottom:1px solid {T['border']};
            padding:4px 24px;display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:0.68rem;color:#5a6478;font-family:monospace;">GRSD v2.0 · Live yfinance Feed</span>
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:0.68rem;color:{T['muted']};font-family:monospace;">
      📅 {live_timestamp(_active_country)}
    </span>
    <span style="font-size:0.62rem;background:rgba(0,200,83,0.15);color:#00c853;
                 border:1px solid rgba(0,200,83,0.35);border-radius:4px;padding:1px 8px;
                 font-family:monospace;">● Live (~15 min delay)</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:14px 0 20px 0;border-bottom:1px solid {T['border']};margin-bottom:18px;">
  <div>
    <div style="font-family:monospace;font-size:1.45rem;font-weight:600;color:{T['bright']};letter-spacing:0.02em;">
      📡 Global RS &amp; Momentum Dashboard
    </div>
    <div style="font-size:0.76rem;color:{T['muted']};margin-top:4px;font-family:'DM Sans',sans-serif;">
      Sector Rotation · Relative Strength · Live NSE + NYSE Data
    </div>
  </div>
  <div style="text-align:right;font-family:monospace;font-size:0.72rem;color:{T['muted']};">
    India NSE &amp; US Equities<br>
    <span style="color:#00c853;">● yfinance Live</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    # Country selector
    st.markdown("### 🌐 Market")
    country_col1, country_col2 = st.columns(2)
    with country_col1:
        india_btn = st.button("🇮🇳 India", use_container_width=True,
                              type="primary" if _active_country == "INDIA" else "secondary")
    with country_col2:
        us_btn = st.button("🇺🇸 USA", use_container_width=True,
                           type="primary" if _active_country == "US" else "secondary")

    if india_btn:
        st.session_state["active_country"] = "INDIA"
        st.rerun()
    if us_btn:
        st.session_state["active_country"] = "US"
        st.rerun()

    market = st.session_state.get("active_country", "INDIA")

    st.markdown("---")
    min_prob = st.slider("Min Probability Score", 0, 80, 35, 5,
                         help="Filter setups below this threshold")

    st.markdown("---")
    ta_ticker = st.text_input(
        "Technical Analysis Ticker",
        value="RELIANCE.NS" if market == "INDIA" else "AAPL",
        placeholder="TCS.NS / AAPL",
    )
    ta_lookback = st.slider("Chart Lookback (Days)", 30, 365, 90, 15)

    st.markdown("---")
    rrg_stage_placeholder = st.empty()

    st.markdown("---")
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"""
<div style="font-size:0.66rem;color:{T['muted']};margin-top:16px;line-height:1.6;font-family:monospace;">
Data: Yahoo Finance (yfinance)<br>
Delay: ~15 min (free feed)<br>
Cache TTL: 15 minutes<br>
RRG: JdK RS-Ratio method
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD LIVE DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("📡 Fetching live market data..."):
    # Benchmarks
    india_bench = fetch_single(INDIA_BENCHMARK, period="1y")
    us_bench    = fetch_single(US_BENCHMARK,    period="1y")

    # Sector indices / ETFs
    india_sect_prices = fetch_sector_indices("INDIA")
    us_sect_prices    = fetch_sector_indices("US")

    # Active market data
    if market == "INDIA":
        bench_df     = india_bench
        sect_prices  = india_sect_prices
        sect_labels  = all_sector_labels("INDIA")
        bench_label  = "Nifty 50"
        bench_ticker = INDIA_BENCHMARK
    else:
        bench_df     = us_bench
        sect_prices  = us_sect_prices
        sect_labels  = all_sector_labels("US")
        bench_label  = "S&P 500"
        bench_ticker = US_BENCHMARK

    # RRG for both markets
    india_rrg = compute_rrg(india_sect_prices, india_bench)
    us_rrg    = compute_rrg(us_sect_prices,    us_bench)
    active_rrg = india_rrg if market == "INDIA" else us_rrg

# Sidebar sector stages
with rrg_stage_placeholder.container():
    render_sidebar_sectors(active_rrg, sect_labels)


# ─────────────────────────────────────────────────────────────────────────────
# RELATIVE PERFORMANCE (shared)
# ─────────────────────────────────────────────────────────────────────────────

LOOKBACKS = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
rs_df = compute_relative_perf(sect_prices, bench_df, LOOKBACKS)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_perf, tab_heat, tab_setups, tab_ta, tab_global = st.tabs([
    "📈 Performance",
    "🔥 Heatmap",
    "⚡ Trade Setups",
    "📊 Technical Analysis",
    "🌍 Global Themes",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_perf:
    st.markdown(f'<div class="section-hdr">Multi-Timeframe Relative Performance — {market}</div>',
                unsafe_allow_html=True)

    p_col1, p_col2 = st.columns([3, 1])

    with p_col2:
        st.markdown("**Timeframe Selection**")
        selected_periods = st.multiselect(
            "Show Periods",
            list(LOOKBACKS.keys()),
            default=["1M", "3M", "6M"],
        )

    filtered_lb = {k: v for k, v in LOOKBACKS.items() if k in selected_periods}
    if not filtered_lb:
        filtered_lb = {"3M": 63}

    with p_col1:
        render_performance_chart(
            sect_prices, bench_df, sect_labels,
            title=f"{'India Sector' if market == 'INDIA' else 'US Sector ETF'} RS vs {bench_label}",
        )

    st.markdown("---")
    st.markdown(f"**RS Score Table — All {market} Sectors vs {bench_label}**")

    rs_filtered = compute_relative_perf(sect_prices, bench_df, filtered_lb)
    render_rs_table(rs_filtered, sect_labels)

    # Top/Bottom KPIs
    if not rs_df.empty and "3M" in rs_df.columns:
        ranked = rs_df[["3M"]].dropna().sort_values("3M", ascending=False)
        ranked.index = [sect_labels.get(k, k) for k in ranked.index]

        st.markdown("---")
        st.markdown("**3M Leaders vs Laggards**")
        kpi_cols = st.columns(5)
        for i, (idx, row) in enumerate(ranked.head(5).iterrows()):
            with kpi_cols[i]:
                val   = row["3M"]
                color = T["green"] if val > 0 else T["red"]
                st.markdown(f"""
<div style="background:{T['bg1']};border:1px solid {T['border']};border-radius:8px;
            padding:12px;text-align:center;">
  <div style="font-size:0.68rem;color:{T['muted']};">{idx}</div>
  <div style="font-family:monospace;font-size:1.1rem;color:{color};font-weight:700;">{val:+.1f}%</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tab_heat:
    h_col1, h_col2 = st.columns([1, 1])

    with h_col1:
        st.markdown('<div class="section-hdr">Relative Rotation Graph (RRG)</div>',
                    unsafe_allow_html=True)

        rrg_mkt = st.radio("RRG Market", ["INDIA", "US"], horizontal=True, key="rrg_mkt_sel")
        rrg_show = india_rrg if rrg_mkt == "INDIA" else us_rrg
        rrg_lbls = all_sector_labels(rrg_mkt)

        if not rrg_show.empty:
            rrg_display = rrg_show.copy()
            rrg_display["sector"] = rrg_display["sector"].apply(
                lambda x: rrg_lbls.get(x, x)[:8]
            )
            render_rrg_chart(rrg_display, rrg_lbls,
                             title=f"RRG — {'India Sectors' if rrg_mkt == 'INDIA' else 'US Sector ETFs'}")
        else:
            st.warning("RRG computing... please wait or refresh.")

        # Stage count badges
        if not rrg_show.empty:
            sc_cols = st.columns(4)
            for col, stage in zip(sc_cols, ["Leading","Weakening","Lagging","Improving"]):
                cnt   = len(rrg_show[rrg_show["stage"] == stage])
                color = RRG_STAGE_COLORS[stage]
                with col:
                    st.markdown(f"""
<div style="background:{color}11;border:1px solid {color}44;border-radius:8px;
            padding:9px;text-align:center;">
  <div style="font-size:0.66rem;color:{color};">{STAGE_EMOJI[stage]} {stage}</div>
  <div style="font-family:monospace;font-size:1.4rem;color:{color};font-weight:800;">{cnt}</div>
</div>""", unsafe_allow_html=True)

    with h_col2:
        st.markdown('<div class="section-hdr">Sector Momentum Heatmap</div>',
                    unsafe_allow_html=True)

        heat_sector = st.selectbox(
            "Select Sector",
            list(INDIA_SECTORS.keys()) if market == "INDIA" else list(US_SECTORS.keys()),
            key="heat_sec",
        )

        with st.spinner("Loading sector stocks..."):
            heat_stocks = fetch_sector_stocks(heat_sector, market)
            bench_c     = _safe_close(bench_df) if "Close" in bench_df.columns else (bench_df.iloc[:,0] if not bench_df.empty else bench_df["Close"])

        heat_rows = []
        for tick, sdf in heat_stocks.items():
            if sdf.empty or len(sdf) < 60:
                continue
            ind = all_indicators(sdf)
            if not ind:
                continue
            ms  = momentum_score(ind)
            rs  = rs_score(_safe_close(sdf), bench_c)
            heat_rows.append({
                "sector":  sect_labels.get(heat_sector, heat_sector),
                "ticker":  tick.replace(".NS", ""),
                "score":   ms,
                "rs":      rs,
                "vol":     ind.get("vol_ratio", 1.0),
                "rsi":     ind.get("rsi14", 50),
                "adx":     ind.get("adx", 0),
            })

        if heat_rows:
            hdf = pd.DataFrame(heat_rows)
            render_heatmap(hdf, f"Momentum Heatmap — {sect_labels.get(heat_sector, heat_sector)}")

            st.markdown("**Live Stock Rankings**")
            display_cols = ["ticker", "score", "rs", "vol", "rsi", "adx"]
            rank_df = hdf[display_cols].sort_values("score", ascending=False)
            rank_df.columns = ["Ticker", "Momentum", "RS%", "Vol×", "RSI", "ADX"]

            def style_m(v):
                if v >= 70: return "color:#00c853;font-weight:bold;"
                if v >= 50: return "color:#69f0ae;"
                if v >= 30: return "color:#ff9800;"
                return "color:#f44336;"

            styled = (rank_df.style
                      .map(style_m, subset=["Momentum"])
                      .format({"Momentum": "{:.0f}", "RS%": "{:+.1f}",
                               "Vol×": "{:.1f}", "RSI": "{:.0f}", "ADX": "{:.0f}"}))
            st.dataframe(styled, use_container_width=True, height=300, hide_index=True)
        else:
            st.info("Loading stock data for heatmap...")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE SETUPS
# ══════════════════════════════════════════════════════════════════════════════

with tab_setups:
    st.markdown('<div class="section-hdr">Live Trade Setup Engine — Automated Scanner</div>',
                unsafe_allow_html=True)

    # Horizon toggle
    hz1, hz2, hz3, hz_sp = st.columns([1, 1, 1, 3])
    with hz1: swing_btn = st.button("⚡ SWING",       use_container_width=True, key="btn_sw")
    with hz2: short_btn = st.button("📈 SHORT TERM",  use_container_width=True, key="btn_st")
    with hz3: long_btn  = st.button("🏦 LONG TERM",   use_container_width=True, key="btn_lt")

    if "horizon" not in st.session_state:
        st.session_state["horizon"] = "SWING"
    if swing_btn: st.session_state["horizon"] = "SWING"
    if short_btn: st.session_state["horizon"] = "SHORT_TERM"
    if long_btn:  st.session_state["horizon"] = "LONG_TERM"

    horizon    = st.session_state["horizon"]
    hz_colors  = {"SWING": "#64b5f6", "SHORT_TERM": "#69f0ae", "LONG_TERM": "#ffd700"}
    hz_labels  = {"SWING": "⚡ SWING (2-10 Days)",
                  "SHORT_TERM": "📈 SHORT TERM (2-6 Weeks)",
                  "LONG_TERM":  "🏦 LONG TERM (3-12 Months)"}

    hzc = hz_colors[horizon]
    st.markdown(f"""
<div style="background:{T['bg1']};border:1px solid {hzc}44;border-radius:8px;
            padding:11px 16px;margin:10px 0;display:flex;align-items:center;justify-content:space-between;">
  <span style="font-family:monospace;font-weight:700;font-size:0.88rem;color:{hzc};">
    Active: {hz_labels[horizon]}
  </span>
  <span style="font-size:0.72rem;color:{T['muted']};">Min Score: {min_prob}</span>
</div>""", unsafe_allow_html=True)

    # Market selector
    scan_mkt = st.radio("Scan Market", ["INDIA", "US"], horizontal=True, key="scan_mkt")

    run_scan = st.button("🔍 Run Live Scanner", type="primary")

    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = pd.DataFrame()

    if run_scan:
        with st.spinner(f"🔍 Scanning live {scan_mkt} data for {horizon} setups..."):
            universe_data = fetch_full_universe(scan_mkt)
            scan_bench    = india_bench if scan_mkt == "INDIA" else us_bench
            scan_rrg      = india_rrg   if scan_mkt == "INDIA" else us_rrg

            results = scan_universe(
                universe_data, scan_bench, scan_rrg,
                horizon, scan_mkt, min_score=min_prob,
            )
            st.session_state["scan_results"] = results
            st.session_state["scan_mkt"]     = scan_mkt

    results_df = st.session_state.get("scan_results", pd.DataFrame())

    if results_df.empty:
        st.markdown(f"""
<div style="background:{T['bg1']};border:1px dashed {T['border']};border-radius:10px;
            padding:48px;text-align:center;color:{T['muted']};">
  <div style="font-size:2.5rem;margin-bottom:12px;">🔍</div>
  <div style="font-size:0.9rem;">Select a horizon and click <strong style="color:{T['bright']};">Run Live Scanner</strong></div>
  <div style="font-size:0.75rem;margin-top:8px;">Scans live OHLCV data across all sectors and ranks by probability score</div>
</div>""", unsafe_allow_html=True)
    else:
        # KPI row
        n_total   = len(results_df)
        n_strong  = len(results_df[results_df["status"] == "Strong Buy"])
        n_watch   = len(results_df[results_df["status"] == "Watch"])
        avg_score = results_df["prob_score"].mean()

        ks1, ks2, ks3, ks4 = st.columns(4)
        for col, label, val, color in [
            (ks1, "Total Setups",    str(n_total),          T["bright"]),
            (ks2, "Strong Buy",      str(n_strong),         T["green"]),
            (ks3, "Watch List",      str(n_watch),          T["gold"]),
            (ks4, "Avg Probability", f"{avg_score:.0f}%",   T["accent"]),
        ]:
            with col:
                st.markdown(f"""
<div style="background:{T['bg1']};border:1px solid {T['border']};border-radius:8px;padding:12px;text-align:center;">
  <div style="font-size:0.66rem;color:{T['muted']};">{label}</div>
  <div style="font-family:monospace;font-size:1.3rem;color:{color};font-weight:700;">{val}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # Sector filter chips
        sectors_avail = sorted(results_df["sector_label"].unique().tolist())
        filter_secs   = st.multiselect("Filter by Sector", sectors_avail,
                                        default=sectors_avail, key="sec_filter")
        filtered_df = results_df[results_df["sector_label"].isin(filter_secs)]

        st.markdown(f"**{len(filtered_df)} setups** — ranked by live probability score")
        st.markdown("---")

        # Cards in 2-column grid
        for chunk_start in range(0, len(filtered_df), 2):
            chunk = filtered_df.iloc[chunk_start:chunk_start + 2]
            cols  = st.columns(2)
            for j, (_, setup) in enumerate(chunk.iterrows()):
                with cols[j]:
                    render_trade_card(setup.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_ta:
    st.markdown('<div class="section-hdr">Technical Analysis Panel — Live Data</div>',
                unsafe_allow_html=True)

    ta_mkt = st.radio("Market", ["INDIA", "US"], horizontal=True, key="ta_mkt_sel",
                      index=0 if market == "INDIA" else 1)
    ticker_input = st.text_input(
        "Enter Ticker Symbol",
        value=ta_ticker,
        placeholder="RELIANCE.NS / TCS.NS / AAPL / MSFT",
        key="ta_tick",
    )

    if ticker_input:
        with st.spinner(f"Loading live data for {ticker_input}..."):
            ta_df    = fetch_single(ticker_input.upper().strip(), period="1y")
            ta_bench = india_bench if ta_mkt == "INDIA" else us_bench

        if ta_df.empty:
            st.error(f"Could not load data for **{ticker_input}**. Check the ticker symbol (NSE stocks need `.NS` suffix).")
        else:
            ind     = all_indicators(ta_df)
            ta_close = ta_df["Close"] if "Close" in ta_df.columns else pd.Series(dtype=float)
            ta_bench_close = ta_bench["Close"] if "Close" in ta_bench.columns else pd.Series(dtype=float)
            rs_val  = rs_score(ta_close, ta_bench_close)
            ind["rs_value"] = rs_val

            close   = ind.get("close", 0)
            ema9v   = ind.get("ema9",  0)
            ema20v  = ind.get("ema20", 0)
            ema50v  = ind.get("ema50", 0)
            adx_v   = ind.get("adx",  0)
            rsi_v   = ind.get("rsi14",0)
            vol_r   = ind.get("vol_ratio", 1.0)
            smi_v   = ind.get("smi",  0)
            atr_v   = ind.get("atr14",0)

            # No-Trade Zone
            ntz = adx_v < 20 or (close > ema9v * 0.995 and close < ema20v * 1.005)
            if ntz:
                st.markdown(
                    '<div class="ntz-warn">⚠️ NO-TRADE ZONE: ADX &lt; 20 or Price between EMA9–EMA21 — choppy conditions</div>',
                    unsafe_allow_html=True,
                )

            # KPI strip
            kpi_data = [
                ("Close",  f"{close:,.2f}",      T["bright"]),
                ("EMA 9",  f"{ema9v:,.2f}",      "#64b5f6" if close > ema9v  else T["red"]),
                ("EMA 20", f"{ema20v:,.2f}",     "#64b5f6" if close > ema20v else T["red"]),
                ("EMA 50", f"{ema50v:,.2f}",     "#64b5f6" if close > ema50v else T["red"]),
                ("ADX",    f"{adx_v:.1f}",       T["green"] if adx_v > 25 else ("#ffd740" if adx_v > 20 else T["red"])),
                ("RSI 14", f"{rsi_v:.1f}",       T["red"] if rsi_v > 68 else ("#ffd740" if rsi_v > 60 else T["green"])),
                ("Vol ×",  f"{vol_r:.1f}×",      T["green"] if vol_r > 1.5 else ("#ffd740" if vol_r > 1 else T["red"])),
                ("SMI",    f"{smi_v:.1f}",       T["green"] if smi_v > 0 else T["red"]),
            ]
            kpi_cols = st.columns(8)
            for (label, val, color), col in zip(kpi_data, kpi_cols):
                with col:
                    st.markdown(f"""
<div style="background:{T['bg1']};border:1px solid {T['border']};border-radius:8px;padding:10px;text-align:center;">
  <div style="font-size:0.63rem;color:{T['muted']};">{label}</div>
  <div style="font-family:monospace;font-size:0.88rem;color:{color};font-weight:700;">{val}</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            # Main chart
            render_candlestick(ta_df, ticker_input.upper(), ind, ta_lookback)

            # Detail panels
            st.markdown("---")
            d_col1, d_col2 = st.columns(2)

            with d_col1:
                st.markdown("**Pivot Support & Resistance**")
                sr_data = {
                    "Level": ["R2", "R1", "Close", "S1", "S2"],
                    "Price":  [ind.get("R2",0), ind.get("R1",0), close,
                               ind.get("S1",0), ind.get("S2",0)],
                    "Type":   ["Resistance","Resistance","Current","Support","Support"],
                }
                sr_df = pd.DataFrame(sr_data)

                def sr_color(row):
                    if row["Type"] == "Resistance": return ["color:#f44336"]*3
                    if row["Type"] == "Support":    return ["color:#00c853"]*3
                    return ["color:#e6edf3"]*3

                styled_sr = sr_df.style.apply(sr_color, axis=1).format({"Price": "{:,.2f}"})
                st.dataframe(styled_sr, hide_index=True, use_container_width=True)

                st.markdown(f"""
<div style="font-family:monospace;font-size:0.78rem;color:{T['muted']};margin-top:8px;">
  ATR(14): {atr_v:,.2f} &nbsp;|&nbsp;
  2x ATR Stop: {(close - 2*atr_v):,.2f} &nbsp;|&nbsp;
  RS vs Bench: <span style="color:{'#00c853' if rs_val>0 else '#f44336'};">{rs_val:+.2f}%</span>
</div>""", unsafe_allow_html=True)

            with d_col2:
                st.markdown("**52-Week Position**")
                from ui_components import w52_bar_html
                st.markdown(w52_bar_html(
                    ind.get("position_pct", 50),
                    ind.get("pct_from_high", 0),
                ), unsafe_allow_html=True)

                h52 = ind.get("high52", 0)
                l52 = ind.get("low52",  0)
                currency = "₹" if ta_mkt == "INDIA" else "$"
                st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:10px;">
  <div style="text-align:center;">
    <div style="font-size:0.65rem;color:{T['muted']};">52W High</div>
    <div style="font-family:monospace;color:{T['gold']};font-size:0.84rem;font-weight:700;">{currency}{h52:,.2f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:0.65rem;color:{T['muted']};">Current</div>
    <div style="font-family:monospace;color:{T['bright']};font-size:0.84rem;font-weight:700;">{currency}{close:,.2f}</div>
  </div>
  <div style="text-align:center;">
    <div style="font-size:0.65rem;color:{T['muted']};">52W Low</div>
    <div style="font-family:monospace;color:{T['green']};font-size:0.84rem;font-weight:700;">{currency}{l52:,.2f}</div>
  </div>
</div>""", unsafe_allow_html=True)

            # Signal summary
            st.markdown("---")
            sig_items = [
                ("ADX > 25",   "Trending"  if adx_v > 25 else "Weak",    T["green"] if adx_v > 25 else T["red"]),
                ("EMA9 Cross", "Bullish"   if close > ema9v  else "Below", T["green"] if close > ema9v  else T["red"]),
                ("EMA20",      "Above"     if close > ema20v else "Below", T["green"] if close > ema20v else T["red"]),
                ("EMA50",      "Above"     if close > ema50v else "Below", T["green"] if close > ema50v else T["red"]),
                ("RSI Zone",   "Overbought" if rsi_v>68 else ("Neutral" if rsi_v>50 else "Oversold"),
                 T["red"] if rsi_v>68 else (T["green"] if rsi_v>45 else T["orange"])),
                ("Volume",     f"{vol_r:.1f}x Avg", T["green"] if vol_r>1.5 else (T["gold"] if vol_r>1 else T["red"])),
                ("RS vs Index",f"{rs_val:+.1f}%",   T["green"] if rs_val>0 else T["red"]),
                ("MACD Hist",  f"{ind.get('macd_hist',0):+.3f}",
                 T["green"] if ind.get("macd_hist",0)>0 else T["red"]),
            ]
            sig_cols = st.columns(4)
            for i, (k, v, c) in enumerate(sig_items):
                with sig_cols[i % 4]:
                    st.markdown(f"""
<div style="background:{T['bg1']};border:1px solid {T['border']};border-radius:8px;padding:9px;margin-bottom:8px;">
  <div style="font-size:0.63rem;color:{T['muted']};">{k}</div>
  <div style="font-family:monospace;font-size:0.82rem;color:{c};font-weight:700;">{v}</div>
</div>""", unsafe_allow_html=True)

    else:
        st.info("Enter a ticker symbol above to load the live technical analysis panel.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GLOBAL THEMES
# ══════════════════════════════════════════════════════════════════════════════

with tab_global:
    st.markdown('<div class="section-hdr">Global Market Themes — Live US + India Synchronization</div>',
                unsafe_allow_html=True)

    # Build stage maps from live RRG
    india_stages: Dict[str, str] = {}
    if not india_rrg.empty:
        for _, row in india_rrg.iterrows():
            india_stages[row["sector"]] = row["stage"]

    us_stages: Dict[str, str] = {}
    if not us_rrg.empty:
        for _, row in us_rrg.iterrows():
            us_stages[row["sector"]] = row["stage"]

    # Global theme mappings
    GLOBAL_THEMES = [
        ("Technology",          "XLK",  "IT",     "Nifty IT"),
        ("Financials / Banking","XLF",  "BANK",   "Bank Nifty"),
        ("Energy",              "XLE",  "ENERGY", "Nifty Energy"),
        ("Healthcare / Pharma", "XLV",  "PHARMA", "Nifty Pharma"),
        ("Industrials / Infra", "XLI",  "INFRA",  "Nifty Infra"),
        ("Materials / Metal",   "XLB",  "METAL",  "Nifty Metal"),
        ("Consumer Staples",    "XLP",  "FMCG",   "Nifty FMCG"),
        ("Real Estate",         "XLRE", "REALTY", "Nifty Realty"),
        ("Autos / Cons Disc",   "XLY",  "AUTO",   "Nifty Auto"),
    ]

    # Sync score
    synced = sum(
        1 for _, us_etf, ind_sec, _ in GLOBAL_THEMES
        if us_stages.get(us_etf) in ("Leading","Improving")
        and india_stages.get(ind_sec) in ("Leading","Improving")
    )
    st.markdown(f"""
<div class="sync-banner">
  <div style="font-family:monospace;font-weight:700;color:{T['accent']};font-size:0.9rem;">
    🌐 Live Global Synchronization: {synced}/{len(GLOBAL_THEMES)} themes aligned
  </div>
  <div style="font-size:0.74rem;color:{T['muted']};margin-top:4px;">
    Real-time RRG alignment across US SPDR ETFs and Indian sector indices
  </div>
</div>""", unsafe_allow_html=True)

    # Theme cards — 2 per row
    for chunk_start in range(0, len(GLOBAL_THEMES), 2):
        chunk = GLOBAL_THEMES[chunk_start:chunk_start + 2]
        tc    = st.columns(2)
        for j, (theme, us_etf, ind_sec, ind_label) in enumerate(chunk):
            us_stg  = us_stages.get(us_etf,  "Unknown")
            ind_stg = india_stages.get(ind_sec, "Unknown")
            both_lead = (us_stg == "Leading" and ind_stg == "Leading")
            both_bull  = us_stg in ("Leading","Improving") and ind_stg in ("Leading","Improving")
            if both_lead:
                signal = "Live: Strong global conviction — both markets in sync"
            elif both_bull:
                signal = "Live: Positive alignment — one or both improving"
            elif us_stg == "Lagging" and ind_stg == "Lagging":
                signal = "Live: Both markets weak — avoid this theme"
            else:
                signal = f"Live: Divergent — US={us_stg}, India={ind_stg}"
            with tc[j]:
                render_theme_card(theme, us_etf, ind_label, us_stg, ind_stg, signal)

    # US ETF live RS table
    st.markdown("---")
    st.markdown("**Live US Sector ETF Performance vs S&P 500**")
    us_rs_rows = []
    for etf_key in US_SECTORS:
        if etf_key in us_sect_prices and not us_sect_prices[etf_key].empty:
            r = rs_score(_safe_close(us_sect_prices[etf_key]), _safe_close(us_bench))
            s = us_stages.get(etf_key, "Unknown")
            us_rs_rows.append({"ETF": etf_key, "Sector": US_SECTORS[etf_key]["label"],
                                "RS% (3M)": r, "Stage": s})
    if us_rs_rows:
        us_rs_df = pd.DataFrame(us_rs_rows).sort_values("RS% (3M)", ascending=False)
        def clr_rs(v):
            if v > 5:  return "color:#00c853;font-weight:700;"
            if v > 0:  return "color:#69f0ae;"
            if v > -5: return "color:#ff9800;"
            return "color:#f44336;font-weight:700;"
        def clr_stg(v):
            return f"color:{RRG_STAGE_COLORS.get(v,'#8b949e')};font-weight:700;"
        styled_us = (us_rs_df.style
                     .map(clr_rs,  subset=["RS% (3M)"])
                     .map(clr_stg, subset=["Stage"])
                     .format({"RS% (3M)": "{:+.1f}%"}))
        st.dataframe(styled_us, hide_index=True, use_container_width=True, height=380)

    # India sector live RS table
    st.markdown("---")
    st.markdown("**Live India Sector Performance vs Nifty 50**")
    ind_rs_rows = []
    for sec_key, info in INDIA_SECTORS.items():
        if sec_key in india_sect_prices and not india_sect_prices[sec_key].empty:
            r = rs_score(_safe_close(india_sect_prices[sec_key]), _safe_close(india_bench))
            s = india_stages.get(sec_key, "Unknown")
            ind_rs_rows.append({"Sector": info["label"], "Key": sec_key,
                                 "RS% (3M)": r, "Stage": s})
    if ind_rs_rows:
        ind_rs_df = pd.DataFrame(ind_rs_rows).sort_values("RS% (3M)", ascending=False)
        styled_ind = (ind_rs_df[["Sector","RS% (3M)","Stage"]].style
                      .map(clr_rs,  subset=["RS% (3M)"])
                      .map(clr_stg, subset=["Stage"])
                      .format({"RS% (3M)": "{:+.1f}%"}))
        st.dataframe(styled_ind, hide_index=True, use_container_width=True, height=360)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="margin-top:40px;border-top:1px solid {T['border']};padding-top:14px;
            display:flex;justify-content:space-between;align-items:center;">
  <div style="font-size:0.68rem;color:{T['muted']};font-family:monospace;">
    Global RS and Momentum Dashboard v2.0 &nbsp;|&nbsp; Data: Yahoo Finance (yfinance) ~15 min delay
  </div>
  <div style="font-size:0.68rem;color:{T['muted']};font-family:monospace;">
    Not financial advice. For educational and research use only.
  </div>
</div>
""", unsafe_allow_html=True)
