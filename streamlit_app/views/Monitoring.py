"""
Page 1: Monitoring Live
Real-time macroeconomic situation, quadrant scatter plot, and allocation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_loader import QUADRANT_NAMES, QUADRANT_COLORS, ALLOCATIONS


def render(data):
    st.header("Situation Macroeconomique Actuelle")

    # === 18-Day Trend (Last 18 days) ===
    st.subheader("Tendance Recente (18 derniers jours - Fenetre de Lissage)")
    if data['quadrants'] is not None:
        last_18 = data['quadrants'].tail(18).copy()

        q_counts = last_18['assigned_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)

        fig_trend = go.Figure(data=[go.Bar(
            x=[f"Q{i} {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=q_counts.values,
            textposition='auto',
        )])

        fig_trend.update_layout(
            title="Repartition des Quadrants (Brut) sur 18 jours",
            yaxis_title="Nombre de Jours",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        dominant_q = q_counts.idxmax()
        st.info(f"Le modele selectionne le **Mode (Valeur la plus frequente)** sur 18 jours. Tendance actuelle : **Q{dominant_q} {QUADRANT_NAMES.get(dominant_q)}** avec {q_counts.max()} jours.")

    st.divider()

    # === Full Distribution (All time) ===
    st.subheader("Repartition Globale des Quadrants (Brut)")
    if data['quadrants'] is not None:
        all_data = data['quadrants'].copy()
        q_counts_all = all_data['assigned_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)

        fig_all = go.Figure(data=[go.Bar(
            x=[f"Q{i} {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=q_counts_all.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=q_counts_all.values,
            textposition='auto',
        )])

        fig_all.update_layout(
            title="Repartition des Quadrants (Brut)",
            yaxis_title="Nombre de Jours",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_all, use_container_width=True)

        dominant_q = q_counts_all.idxmax()
        st.info(f"Repartition globale (Brut) : **Q{dominant_q} {QUADRANT_NAMES.get(dominant_q)}** avec {q_counts_all.max()} jours.")

    st.divider()

    # === Smooth Quadrant Distribution (from Backtest) ===
    st.subheader("Repartition des Quadrants Lisses (Backtest Complet)")
    if data['backtest'] is not None and 'smooth_quadrant' in data['backtest'].columns:
        smooth_q_counts = data['backtest']['smooth_quadrant'].value_counts().reindex([1, 2, 3, 4], fill_value=0)
        total_days = smooth_q_counts.sum()

        fig_smooth = go.Figure(data=[go.Bar(
            x=[f"Q{i} {QUADRANT_NAMES.get(i)}" for i in [1, 2, 3, 4]],
            y=smooth_q_counts.values,
            marker_color=[QUADRANT_COLORS[i] for i in [1, 2, 3, 4]],
            text=[f"{v} ({v / total_days * 100:.1f}%)" for v in smooth_q_counts.values],
            textposition='auto',
        )])

        fig_smooth.update_layout(
            title=f"Repartition des Quadrants (Lisse 18j) - {total_days} jours de trading",
            yaxis_title="Nombre de Jours",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_smooth, use_container_width=True)

        dominant_smooth_q = smooth_q_counts.idxmax()
        start_date = data['backtest']['date'].min().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        end_date = data['backtest']['date'].max().strftime('%Y-%m-%d') if 'date' in data['backtest'].columns else 'N/A'
        st.info(f"Periode: **{start_date}** -> **{end_date}** | Regime dominant (lisse): **Q{dominant_smooth_q} {QUADRANT_NAMES.get(dominant_smooth_q)}** ({smooth_q_counts.max() / total_days * 100:.1f}%)")
    else:
        st.warning("Donnees smooth_quadrant non disponibles. Lancez le backtest pour generer ces donnees.")

    st.divider()

    # === Scatter Plot + Allocation ===
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Position dans le Cycle")
        if data['quadrants'] is not None:
            df_q = data['quadrants']

            fig = go.Figure()

            # Quadrant backgrounds
            fig.add_shape(type="rect", x0=-3, y0=0, x1=0, y1=3, fillcolor="rgba(0,255,0,0.15)", line_width=0)
            fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=3, fillcolor="rgba(255,165,0,0.15)", line_width=0)
            fig.add_shape(type="rect", x0=0, y0=-3, x1=3, y1=0, fillcolor="rgba(255,0,0,0.15)", line_width=0)
            fig.add_shape(type="rect", x0=-3, y0=-3, x1=0, y1=0, fillcolor="rgba(0,0,255,0.15)", line_width=0)

            if 'MACRO_GROWTH_SCORE' in df_q.columns and 'MACRO_INFLATION_SCORE' in df_q.columns:
                inflation_hist = df_q['MACRO_INFLATION_SCORE']
                growth_hist = df_q['MACRO_GROWTH_SCORE']
                latest = df_q.iloc[-1]
                cur_inflation = latest['MACRO_INFLATION_SCORE']
                cur_growth = latest['MACRO_GROWTH_SCORE']
                x_title = "Weighted Inflation Score (New Logic) ->"
                y_title = "Weighted Growth Score (New Logic) ->"
                st.caption("Affichage base sur la nouvelle logique **2-Axes (Moyenne Ponderee)**")
            else:
                inflation_hist = df_q['score_Q2'] - df_q['score_Q4']
                growth_hist = df_q['score_Q1'] - df_q['score_Q3']
                latest = df_q.iloc[-1]
                cur_inflation = latest['score_Q2'] - latest['score_Q4']
                cur_growth = latest['score_Q1'] - latest['score_Q3']
                x_title = "Score d'Inflation ->"
                y_title = "Score de Croissance ->"

            fig.add_trace(go.Scatter(
                x=inflation_hist, y=growth_hist, mode='markers',
                marker=dict(size=4, color=list(range(len(df_q))), colorscale='Blues', opacity=0.4, showscale=False),
                name='Historique complet',
                hovertext=df_q['date'].dt.strftime('%Y-%m-%d') if 'date' in df_q.columns else None
            ))

            df_recent = df_q.tail(90)
            if 'MACRO_GROWTH_SCORE' in df_recent.columns:
                fig.add_trace(go.Scatter(
                    x=df_recent['MACRO_INFLATION_SCORE'], y=df_recent['MACRO_GROWTH_SCORE'],
                    mode='markers',
                    marker=dict(size=6, color='yellow', opacity=0.8, line=dict(width=0.5, color='black')),
                    name='90 derniers jours'
                ))

            fig.add_trace(go.Scatter(
                x=[cur_inflation], y=[cur_growth], mode='markers',
                marker=dict(size=20, color='red', symbol='star'), name='Actuel'
            ))

            fig.update_layout(
                xaxis_title=x_title, yaxis_title=y_title, height=400, showlegend=True,
                xaxis=dict(range=[-2.5, 2.5]), yaxis=dict(range=[-2.5, 2.5])
            )

            fig.add_annotation(x=-1.2, y=1.2, text="Q1: Croissance", showarrow=False, font=dict(size=11))
            fig.add_annotation(x=1.2, y=1.2, text="Q2: Inflation", showarrow=False, font=dict(size=11))
            fig.add_annotation(x=1.2, y=-1.2, text="Q3: Stagflation", showarrow=False, font=dict(size=11))
            fig.add_annotation(x=-1.2, y=-1.2, text="Q4: Deflation", showarrow=False, font=dict(size=11))

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Donnees quadrants non disponibles")

    with c2:
        st.subheader("Allocation Actuelle")

        if data['backtest'] is not None:
            current_bt_q = int(data['backtest'].iloc[-1].get('smooth_quadrant', 1))
            alloc = ALLOCATIONS.get(current_bt_q, ALLOCATIONS[1])
            st.caption(f"Base sur le **Regime Modele Q{current_bt_q}** (Lisse)")
            alloc_df = pd.DataFrame({'Asset': alloc.keys(), 'Weight': alloc.values()})
            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        elif data['quadrants'] is not None:
            current_q = int(data['quadrants'].iloc[-1].get('assigned_quadrant', 1))
            alloc = ALLOCATIONS.get(current_q, ALLOCATIONS[1])
            st.warning(f"Base sur le Regime Brut Q{current_q} (Donnees Backtest manquantes)")
            alloc_df = pd.DataFrame({'Asset': alloc.keys(), 'Weight': alloc.values()})
            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Allocation non disponible")

    # === Recent Regime History ===
    st.subheader("Historique recent des Regimes")
    if data['quadrants'] is not None:
        recent = data['quadrants'].tail(20)[['date', 'assigned_quadrant', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']]
        recent['Regime'] = recent['assigned_quadrant'].map(QUADRANT_NAMES)
        st.dataframe(recent[['date', 'Regime', 'score_Q1', 'score_Q2', 'score_Q3', 'score_Q4']], use_container_width=True)
