"""
US Macro Four Seasons Strategy - Main Dashboard
Single-page app with tab navigation (no sidebar pages).

Run: streamlit run streamlit_app/app.py
"""

import sys
import os
import streamlit as st
import pandas as pd

# Add streamlit_app to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, apply_theme, render_sidebar, QUADRANT_NAMES
from views import Monitoring, Backtest, ML_Performance, Methodologie, Correlation_Indicators

# --- PAGE CONFIG (only once) ---
st.set_page_config(
    layout="wide",
    page_title="US Macro Strategy",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

# --- THEME & DATA ---
apply_theme()
data = load_data()
render_sidebar(data)

# --- HEADER KPIs ---
st.title("US Macro Four Seasons Strategy")
st.markdown("""
**Objectif :** Surperformer le marche en adaptant l'allocation d'actifs selon le cycle economique US.
**Approche :** Modele 4 Quadrants base sur les taux de changement de la **Croissance** et de l'**Inflation**.
""")

c_header1, c_header2, c_header3, c_header4, c_header5 = st.columns(5)

if data['quadrants'] is not None:
    latest = data['quadrants'].iloc[-1]

    raw_q = int(latest.get('assigned_quadrant', 0))
    c_header1.metric("Raw Q (Today)", f"Q{raw_q} {QUADRANT_NAMES.get(raw_q, 'Unknown')}", delta="Volatile Signal", delta_color="off")

    if data['backtest'] is not None:
        latest_bt = data['backtest'].iloc[-1]
        smooth_q = int(latest_bt.get('smooth_quadrant', 0))
        c_header2.metric("Model Q (Allocated)", f"Q{smooth_q} {QUADRANT_NAMES.get(smooth_q, 'Unknown')}", delta="Stable Signal", delta_color="normal")
    else:
        c_header2.metric("Model Q", "N/A")

    # RISK REGIME (was Growth)
    # PROB_GROWTH_EMA is now PROB_RISK_ON (Spread Trend)
    risk_prob = latest.get('PROB_GROWTH_EMA', latest.get('score_Q1', 0))
    if 'PROB_GROWTH_EMA' in latest:
        risk_label = "Risk On" if risk_prob > 0.5 else "Risk Off"
        c_header3.metric("Risk Regime", f"{risk_label} ({risk_prob:.0%})")
    else:
        # Legacy fallback
        c_header3.metric("Risk Regime", f"{risk_prob:.2f}")

    # RATES REGIME (was Inflation)
    # PROB_INFLATION_EMA is now PROB_REFLATION (Breakeven Trend)
    rates_prob = latest.get('PROB_INFLATION_EMA', latest.get('score_Q2', 0))
    if 'PROB_INFLATION_EMA' in latest:
        rates_label = "Reflation" if rates_prob > 0.5 else "Disinflation"
        c_header4.metric("Rates Regime", f"{rates_label} ({rates_prob:.0%})")
    else:
        # Legacy fallback
        c_header4.metric("Rates Regime", f"{rates_prob:.2f}")

    last_date = latest['date'].strftime("%Y-%m-%d") if pd.notnull(latest['date']) else "N/A"
    c_header5.metric("Last Update", last_date)

st.divider()

# --- TAB NAVIGATION (no emojis, side by side) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Monitoring",
    "Backtest & Perf",
    "ML Performance",
    "Methodologie",
    "Correlation & Indicators"
])

with tab1:
    Monitoring.render(data)

with tab2:
    Backtest.render(data)

with tab3:
    ML_Performance.render(data)

with tab4:
    Methodologie.render(data)

with tab5:
    Correlation_Indicators.render(data)
