"""
Page 2: Backtest & Performance
Historical strategy performance, heatmaps by quadrant regime.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render(data):
    st.header("Performance Historique")

    # === Strategy vs Benchmark ===
    st.subheader("Strategie vs Benchmark")
    if data['backtest'] is not None:
        df_bt = data['backtest']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['wealth'], name='Strategy', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['SP500_wealth'], name='SP500', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['GOLD_wealth'], name='Gold', line=dict(color='gold')))
        fig.update_layout(height=400, yaxis_title="Wealth ($)", xaxis_title="Date")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Donnees backtest non disponibles")

    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    if data['stats'] is not None:
        stats_dict = data['stats'].iloc[0].to_dict() if len(data['stats']) > 0 else {}
        m1.metric("Total Return", f"{stats_dict.get('total_return', 0) * 100:.1f}%")
        m2.metric("Max Drawdown", f"{stats_dict.get('strategy_max_drawdown', 0) * 100:.1f}%")
        m3.metric("Sharpe Ratio", f"{stats_dict.get('strategy_sharpe_annual', 0):.2f}")
        m4.metric("Annual Vol", f"{stats_dict.get('strategy_vol_annual', 0) * 100:.1f}%")

    st.divider()

    # === Helper for heatmap rendering ===
    def _render_heatmap(perf_data, title, caption, height=350):
        if perf_data is not None and 'sharpe' in perf_data.columns:
            df = perf_data.copy()
            # Pivot using 'sharpe' instead of 'annual_return'
            pivot = df.pivot(index='asset', columns='quadrant', values='sharpe').fillna(0)
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,  # Sharpe is already a ratio, no need to multiply by 100
                x=[f'Q{int(c)}' for c in pivot.columns],
                y=pivot.index,
                colorscale='RdYlGn',
                text=[[f'{v:.2f}' for v in row] for row in pivot.values],  # Format as 2 decimals
                texttemplate='%{text}',
                textfont={"size": 10},
                zmid=0
            ))
            fig.update_layout(height=height, title=title)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(caption)
            return True
        return False

    # =========================================================
    # SECTION 1: Performance avec Quadrants PREDITS (ML)
    # =========================================================
    st.subheader("Performance par Quadrant PREDIT (Modele ML)")
    st.markdown("*Ratio de Sharpe par actif en utilisant les quadrants assignes par le modele ML (predict_proba + EMA).*")

    # Heatmap 1: Actions ETF — Predicted
    perf_source = data.get('perf_smooth') if data.get('perf_smooth') is not None else data.get('perf')
    if not _render_heatmap(
        perf_source,
        "Actions/ETF — Sharpe Ratio (Quadrants Predits ML)",
        "Quadrants ML : Random Forest predict_proba + EMA span=5."
    ):
        st.info("Donnees de performance Actions non disponibles.")

    # Heatmap 2: Forex — Predicted
    if not _render_heatmap(
        data.get('forex_perf'),
        "Forex — Sharpe Ratio (Quadrants Predits ML)",
        "Paires Forex (USD/XXX et Inverses XXX/USD avec Carry) — quadrants ML."
    ):
        st.info("Donnees de performance Forex non disponibles.")

    st.divider()

    # =========================================================
    # SECTION 2: Performance avec Quadrants TARGET (Ground Truth)
    # =========================================================
    st.subheader("Performance par Quadrant TARGET (Ground Truth)")
    st.markdown("*Ratio de Sharpe si le modele etait 100% precis — base sur les quadrants reels (Initial Claims + CPI vs rolling median).*")

    # Heatmap 3: Actions ETF — Target
    if not _render_heatmap(
        data.get('perf_target'),
        "Actions/ETF — Sharpe Ratio (Quadrants Reels Target)",
        "Ground truth : quadrants bases sur INITIAL_CLAIMS_YOY (Growth inv) et CPI_YOY vs rolling median 5 ans."
    ):
        st.info("Donnees de performance Target Actions non disponibles. Relancez le DAG.")

    # Heatmap 4: Forex — Target
    if not _render_heatmap(
        data.get('forex_perf_target'),
        "Forex — Sharpe Ratio (Quadrants Reels Target)",
        "Paires Forex — si le modele etait parfait (quadrants ground truth)."
    ):
        st.info("Donnees de performance Target Forex non disponibles. Relancez le DAG.")

