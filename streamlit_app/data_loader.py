"""
Shared data loader and configuration for the Streamlit multi-page app.
All pages import from here to avoid code duplication.
"""

import streamlit as st
import pandas as pd
import os
import json


# --- CONSTANTS ---
QUADRANT_NAMES = {1: "Growth", 2: "Inflation", 3: "Stagflation", 4: "Deflation"}
QUADRANT_COLORS = {1: 'green', 2: 'orange', 3: 'red', 4: 'blue'}

ALLOCATIONS = {
    1: {"SP500": 30, "NASDAQ": 40, "SmallCAP": 30},
    2: {"SP500": 40, "GOLD": 30, "COMMODITIES": 20, "NASDAQ": 10},
    3: {"GOLD": 60, "COMMODITIES": 20, "TREASURY": 20},
    4: {"TREASURY": 60, "GOLD": 40}
}


# --- DATA LOADING ---
@st.cache_data
def load_data():
    base_dir = os.path.expanduser("~/airflow/data/US")
    data = {}
    
    try:
        data['backtest'] = pd.read_csv(f"{base_dir}/backtest_results/backtest_timeseries.csv", parse_dates=['date'])
    except:
        data['backtest'] = None
    
    try:
        data['stats'] = pd.read_csv(f"{base_dir}/backtest_results/backtest_stats.csv")
    except:
        data['stats'] = None
    
    try:
        data['quadrants'] = pd.read_csv(f"{base_dir}/output_dag/quadrants.csv", parse_dates=['date'])
    except:
        data['quadrants'] = None
    
    try:
        data['perf'] = pd.read_parquet(f"{base_dir}/output_dag/assets_performance_by_quadrant.parquet")
    except:
        data['perf'] = None
        
    try:
        data['perf_smooth'] = pd.read_csv(f"{base_dir}/backtest_results/assets_performance_by_smooth_quadrant.csv")
    except:
        data['perf_smooth'] = None
    
    try:
        data['forex_perf'] = pd.read_parquet(f"{base_dir}/output_dag/forex_performance_by_quadrant.parquet")
    except:
        data['forex_perf'] = None
    
    try:
        data['perf_target'] = pd.read_parquet(f"{base_dir}/output_dag/assets_performance_by_target_quadrant.parquet")
    except:
        data['perf_target'] = None
    
    try:
        data['forex_perf_target'] = pd.read_parquet(f"{base_dir}/output_dag/forex_performance_by_target_quadrant.parquet")
    except:
        data['forex_perf_target'] = None
    
    try:
        with open(f"{base_dir}/output_dag/ml_metrics.json", 'r') as f:
            data['ml_metrics'] = json.load(f)
    except:
        data['ml_metrics'] = None
    
    return data


# --- THEME CSS ---
BLOOMBERG_CSS = """
    <style>
    /* Force dark background everywhere */
    .stApp {
        background-color: #0a0e27 !important;
    }
    .main {
        background-color: #0a0e27 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 900 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d35 0%, #0a0e27 100%) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e8e8e8 !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    /* All text white/light */
    .stMarkdown, .stMarkdown p, .stMarkdown span, p, span, div, label {
        color: #e8e8e8 !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
    }
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Divider */
    hr {
        border-color: #3d4263 !important;
    }
    
    /* Sidebar collapse button  */
    [data-testid="stSidebarCollapseButton"] button {
        width: 40px !important;
        height: 40px !important;
        font-size: 28px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Tab navigation styling - bigger + separated */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px !important;
        border-bottom: 2px solid #3d4263 !important;
        padding-bottom: 0 !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 !important;
        justify-content: center !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 14px 28px !important;
        color: #8890b5 !important;
        border-right: 2px solid #3d4263 !important;
    }
    .stTabs [data-baseweb="tab"]:last-child {
        border-right: none !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.08) !important;
        border-bottom: 3px solid #00d4ff !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    </style>
"""


def apply_theme():
    """Apply the Bloomberg-style dark theme."""
    st.markdown(BLOOMBERG_CSS, unsafe_allow_html=True)


def render_sidebar(data):
    """Render the shared sidebar."""
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg", width=80)
        st.title("🍂 Four Seasons")
        st.caption("Macro Strategy Dashboard")
        
        if st.button("🔄 Recharger les données", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        st.subheader("📊 État du Pipeline")
        
        if data['quadrants'] is not None:
            last_date = data['quadrants']['date'].max()
            st.success(f"✅ Quadrants: {last_date.strftime('%Y-%m-%d')}")
        else:
            st.error("❌ Quadrants: Non disponible")
        
        if data['backtest'] is not None:
            st.success("✅ Backtest: Disponible")
        else:
            st.warning("⚠️ Backtest: Non disponible")
        
        if data['ml_metrics'] is not None:
            st.success("✅ ML Metrics: Disponible")
        else:
            st.warning("⚠️ ML Metrics: Non disponible")
        
        st.divider()
        
        st.subheader("🔧 Actions")
        st.code("airflow dags trigger dag_us_macro", language="bash")
        
        st.divider()
        
        st.caption("Made with ❤️ using Streamlit")
