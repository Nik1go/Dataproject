"""
Page 5: Correlation & Indicators
Visualization of raw indicators and their relationships.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os


@st.cache_data
def load_indicators():
    path = os.path.expanduser("~/airflow/data/US/output_dag/combined_indicators.csv")
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        return df.sort_values('date')
    except:
        return None


def render(data):
    st.header("Correlation & Indicators")

    indicators = load_indicators()

    if indicators is not None:
        # --- PRE-PROCESSING: Calculate Real Rates & Clean Data ---
        # 1. Calculate Real Rates (Fed Rate - CPI YoY)
        if 'TAUX_FED' in indicators.columns and 'INFLATION' in indicators.columns:
            # Assuming INFLATION is CPI Index, need 1-year change
            # However, check if 'INFLATION' is already YoY or Index. 
            # In other scripts: inflation_yoy = calculate_yoy_change(df['INFLATION'], 252)
            # We will replicate this logic locally for visualization
            indicators['INFLATION_YOY'] = indicators['INFLATION'].pct_change(periods=252) * 100
            indicators['REAL_RATES'] = indicators['TAUX_FED'] - indicators['INFLATION_YOY']
        
        # 2. Filter out unnecessary columns (Net Liquidity components + others if needed)
        cols_to_drop = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'INFLATION'] 
        indicators = indicators.drop(columns=[c for c in cols_to_drop if c in indicators.columns], errors='ignore')

        # --- APPLY PUBLICATION LAGS (REALISTIC CORRELATION) ---
        # Same lags as in compute_quadrants.py / train_model.py
        LAGS_TRADING_DAYS = {
            # Real-time market data (Lag 0)
            'WTI_CRUDE_OIL': 0,
            'US_DOLLAR_INDEX': 0,
            'VIX': 0,
            'BREAKEVEN_10Y': 0,
            'High_Yield_Bond_SPREAD': 0,
            '10-2Year_Treasury_Yield_Bond': 0,
            'COPPER': 0,
            'TAUX_FED': 0,
            'NET_LIQUIDITY': 0,
        
            # Monthly economic indicators (typical publication delays)
            'IND_PRODUCTION': 35,
            'HOUSING_PERMITS': 25,
            'CONSUMER_SENTIMENT': 5,
            'INITIAL_CLAIMS': 5,
            'INFLATION_YOY': 30,
            'USPHCI': 60,
            'Real_Gross_Domestic_Product': 60,
        }
        
        st.info(f"Applying publication lags to reflect realistic information availability.")
        
        for col, lag in LAGS_TRADING_DAYS.items():
            if col in indicators.columns:
                indicators[col] = indicators[col].shift(lag)
        
        # Select only numeric columns for correlation
        numeric_df = indicators.select_dtypes(include=['float64', 'int64'])
        
        # --- 1. NET LIQUIDITY (Existing) ---
        st.subheader("Net Liquidity (Fed)")
        if 'NET_LIQUIDITY' in indicators.columns:
            df_nl = indicators[['date', 'NET_LIQUIDITY']].dropna()
            fig_nl = go.Figure()
            fig_nl.add_trace(go.Scatter(
                x=df_nl['date'], y=df_nl['NET_LIQUIDITY'],
                mode='lines', name='Net Liquidity',
                line=dict(color='cyan', width=2)
            ))
            fig_nl.update_layout(height=400, xaxis_title="Date", yaxis_title="Net Liquidity ($)", hovermode='x unified')
            st.plotly_chart(fig_nl, use_container_width=True)
        
        st.divider()
        
        # --- 2. CORRELATION MATRIX ---
        st.subheader("Global Correlation Matrix")
        
        # Filter out date/unnecessary columns if any remain
        corr_matrix = numeric_df.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu', # Red to Blue (Red=Neg, Blue=Pos)
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate="%{text}"
        ))
        fig_corr.update_layout(height=800, title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.divider()

        # --- 3. TARGET DRIVERS (HY SPREAD & BREAKEVEN) ---
        st.subheader("Target Drivers Analysis")
        st.info("Correlation of all indicators with key Targets (over full history).")

        col_risk, col_inf = st.columns(2)
        
        # Risk Driver (HY Spread)
        with col_risk:
            st.markdown("#### vs High Yield Spread (Risk)")
            if 'High_Yield_Bond_SPREAD' in corr_matrix.columns:
                target_corr = corr_matrix['High_Yield_Bond_SPREAD'].drop('High_Yield_Bond_SPREAD').sort_values()
                
                fig_risk = go.Figure(go.Bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    marker=dict(color=target_corr.values, colorscale='RdBu', cmid=0)
                ))
                fig_risk.update_layout(height=600, xaxis_title="Correlation w/ HY Spread")
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.error("High_Yield_Bond_SPREAD not found.")

        # Inflation Driver (Breakeven)
        with col_inf:
            st.markdown("#### vs Breakeven 10Y (Inflation)")
            if 'BREAKEVEN_10Y' in corr_matrix.columns:
                target_corr = corr_matrix['BREAKEVEN_10Y'].drop('BREAKEVEN_10Y').sort_values()
                
                fig_inf = go.Figure(go.Bar(
                    x=target_corr.values,
                    y=target_corr.index,
                    orientation='h',
                    marker=dict(color=target_corr.values, colorscale='RdBu', cmid=0)
                ))
                fig_inf.update_layout(height=600, xaxis_title="Correlation w/ Breakeven")
                st.plotly_chart(fig_inf, use_container_width=True)
            else:
                st.error("BREAKEVEN_10Y not found.")

        st.divider()

        # --- 4. ROLLING CORRELATION TOOL ---
        st.subheader("Rolling Correlation Analysis")
        st.caption("Analyze how the relationship between two assets changes over time (e.g., during crises).")

        col1, col2, col3 = st.columns(3)
        
        all_assets = numeric_df.columns.tolist()
        
        with col1:
            asset_a = st.selectbox("Asset A", options=all_assets, index=all_assets.index('High_Yield_Bond_SPREAD') if 'High_Yield_Bond_SPREAD' in all_assets else 0)
        with col2:
            asset_b = st.selectbox("Asset B", options=all_assets, index=all_assets.index('BREAKEVEN_10Y') if 'BREAKEVEN_10Y' in all_assets else 1)
        with col3:
            window = st.slider("Rolling Window (Days)", min_value=10, max_value=252*2, value=63, step=5)

        if asset_a and asset_b:
            # Calculate Rolling Correlation
            rolling_corr = indicators[asset_a].rolling(window=window).corr(indicators[asset_b])
            
            # Plot
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(
                x=indicators['date'], 
                y=rolling_corr,
                mode='lines',
                name=f"Corr({asset_a}, {asset_b})",
                line=dict(width=2)
            ))
            
            # Add Zero Line
            fig_roll.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
            
            fig_roll.update_layout(
                height=500,
                title=f"{window}-Day Rolling Correlation: {asset_a} vs {asset_b}",
                yaxis_title="Correlation Coefficient",
                xaxis_title="Date",
                yaxis=dict(range=[-1.1, 1.1])
            )
            st.plotly_chart(fig_roll, use_container_width=True)

    else:
        st.error("Impossible de charger combined_indicators.csv. Lancez le DAG.")
        st.code("airflow dags trigger dag_us_macro", language="bash")
