"""
Page 3: ML Performance
Classification model metrics, walk-forward validation, GridSearchCV, confusion matrix.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render(data):
    st.header("Performance du Modele ML")

    if data['ml_metrics'] is not None:
        metrics = data['ml_metrics']

        # Validation Type Badge
        validation_type = metrics.get('validation_type', 'unknown')
        if validation_type == 'walk_forward':
            st.success("**Validation Walk-Forward** - Metriques Out-of-Sample fiables")
        else:
            st.warning("Type de validation inconnu")

        st.divider()

        # ========================================
        # OUT-OF-SAMPLE METRICS
        # ========================================
        st.subheader("Metriques Out-of-Sample (Walk-Forward)")
        st.caption("Train sur le passe, test sur le futur -> Metriques fiables")

        is_classifier = metrics.get('model_type') == 'binary_classification'

        acc_growth_oos = metrics.get('accuracy_growth_out_of_sample', 0)
        acc_inflation_oos = metrics.get('accuracy_inflation_out_of_sample', 0)
        auc_growth_oos = metrics.get('auc_growth_out_of_sample', 0)
        auc_inflation_oos = metrics.get('auc_inflation_out_of_sample', 0)
        prec_growth_oos = metrics.get('precision_growth_out_of_sample', 0)
        prec_inflation_oos = metrics.get('precision_inflation_out_of_sample', 0)
        rec_growth_oos = metrics.get('recall_growth_out_of_sample', 0)
        rec_inflation_oos = metrics.get('recall_inflation_out_of_sample', 0)
        acc_oos = metrics.get('accuracy_out_of_sample', metrics.get('accuracy_score', 0))

        # KPI row
        col1, col2, col3 = st.columns(3)
        col1.metric("Quadrant Accuracy", f"{acc_oos:.1%}")
        if is_classifier:
            col2.metric("Accuracy Risk (Spread)", f"{acc_growth_oos:.1%}")
            col3.metric("Accuracy Rates (Breakeven)", f"{acc_inflation_oos:.1%}")
        else:
            col2.metric("R2 Risk", f"{acc_growth_oos:.1%}")
            col3.metric("R2 Rates", f"{acc_inflation_oos:.1%}")

        # Precision / Recall / AUC table (Classification)
        if is_classifier:
            st.markdown("#### Detail par Modele")
            pr_df = pd.DataFrame({
                'Model': ['Risk (Spread)', 'Rates (Breakeven)'],
                'Accuracy': [f"{acc_growth_oos:.1%}", f"{acc_inflation_oos:.1%}"],
                'Precision': [f"{prec_growth_oos:.1%}", f"{prec_inflation_oos:.1%}"],
                'Recall': [f"{rec_growth_oos:.1%}", f"{rec_inflation_oos:.1%}"],
                'AUC-ROC': [f"{auc_growth_oos:.3f}", f"{auc_inflation_oos:.3f}"]
            })
            st.dataframe(pr_df, use_container_width=True, hide_index=True)

        st.divider()

        # ========================================
        # WALK-FORWARD PER YEAR CHART
        # ========================================
        wf_growth_auc = metrics.get('walk_forward_growth_per_year_auc', {})
        wf_inflation_auc = metrics.get('walk_forward_inflation_per_year_auc', {})
        wf_growth_acc = metrics.get('walk_forward_growth_per_year_accuracy', {})
        wf_inflation_acc = metrics.get('walk_forward_inflation_per_year_accuracy', {})
        wf_growth_legacy = metrics.get('walk_forward_growth_per_year', {})
        wf_inflation_legacy = metrics.get('walk_forward_inflation_per_year', {})

        has_classification = bool(wf_growth_auc)

        if has_classification:
            st.subheader("Walk-Forward AUC-ROC par Annee")
            wf_g = wf_growth_auc
            wf_i = wf_inflation_auc
            metric_label = 'AUC-ROC'
            y_range = [0.3, 1.0]
        elif wf_growth_legacy:
            st.subheader("Walk-Forward R2 par Annee")
            wf_g = wf_growth_legacy
            wf_i = wf_inflation_legacy
            metric_label = 'R2'
            y_range = [-0.5, 1.0]
        else:
            wf_g = wf_i = None
            metric_label = ''
            y_range = [0, 1]

        if wf_g and wf_i:
            years = sorted(set(wf_g.keys()) | set(wf_i.keys()))

            fig_wf = go.Figure()

            fig_wf.add_trace(go.Scatter(
                x=years,
                y=[wf_g.get(y, None) for y in years],
                mode='lines+markers',
                name=f'Risk (Growth) {metric_label}',
                line=dict(color='green', width=2),
                marker=dict(size=8)
            ))

            fig_wf.add_trace(go.Scatter(
                x=years,
                y=[wf_i.get(y, None) for y in years],
                mode='lines+markers',
                name=f'Rates (Inflation) {metric_label}',
                line=dict(color='orange', width=2),
                marker=dict(size=8)
            ))

            if has_classification:
                fig_wf.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5,
                                 annotation_text="Random (0.5)")
            else:
                fig_wf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig_wf.update_layout(
                height=350,
                xaxis_title="Annee",
                yaxis_title=metric_label,
                yaxis=dict(range=y_range),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_wf, use_container_width=True)

            avg_g = sum(wf_g.values()) / len(wf_g) if wf_g else 0
            avg_i = sum(wf_i.values()) / len(wf_i) if wf_i else 0
            st.caption(f"Moyenne Walk-Forward: Growth {metric_label} = {avg_g:.3f} | Inflation {metric_label} = {avg_i:.3f}")

            if has_classification and wf_growth_acc:
                with st.expander("Accuracy par Annee (detail)"):
                    acc_years = sorted(set(wf_growth_acc.keys()) | set(wf_inflation_acc.keys()))
                    acc_df = pd.DataFrame({
                        'Year': acc_years,
                        'Growth Acc': [f"{wf_growth_acc.get(y, 0):.1%}" for y in acc_years],
                        'Inflation Acc': [f"{wf_inflation_acc.get(y, 0):.1%}" for y in acc_years],
                        'Growth AUC': [f"{wf_growth_auc.get(y, 0):.3f}" for y in acc_years],
                        'Inflation AUC': [f"{wf_inflation_auc.get(y, 0):.3f}" for y in acc_years]
                    })
                    st.dataframe(acc_df, use_container_width=True, hide_index=True)
        else:
            st.info("Donnees Walk-Forward par annee non disponibles.")

        st.divider()

        # ========================================
        # TARGET VARIABLES VISUALIZATION
        # ========================================
        st.subheader("Targets a Predire (Risk & Rates Market Regimes)")

        if data['quadrants'] is not None:
            df_quadrants = data['quadrants']

            # Check for new targets (Spreads & Breakevens)
            # compute_quadrants puts original 'High_Yield_Bond_SPREAD' and 'BREAKEVEN_10Y' in df if present
            has_new_targets = 'High_Yield_Bond_SPREAD' in df_quadrants.columns and 'BREAKEVEN_10Y' in df_quadrants.columns
            
            if has_new_targets:
                # Show full history (approx 25 years to cover 2005+)
                df_recent = df_quadrants[df_quadrants['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=25 * 365))]

                fig_targets = go.Figure()

                # Risk: High Yield Spread (Inverted? No, just plot. High Spread = Risk Off)
                # But we predict Risk On. So low spread = Risk On.
                fig_targets.add_trace(go.Scatter(
                    x=df_recent['date'], y=df_recent['High_Yield_Bond_SPREAD'],
                    mode='lines', name='Risk Proxy (HY Bond Spread %)',
                    line=dict(color='green', width=2)
                ))

                # Inflation: Breakeven
                fig_targets.add_trace(go.Scatter(
                    x=df_recent['date'], y=df_recent['BREAKEVEN_10Y'],
                    mode='lines', name='Inflation Proxy (10Y Breakeven %)',
                    line=dict(color='orange', width=2)
                ))

                # fig_targets.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig_targets.update_layout(
                    height=350, xaxis_title="Date", yaxis_title="Yield / Rate (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig_targets, use_container_width=True)

                col_g, col_i = st.columns(2)
                with col_g:
                    spread_std = df_quadrants['High_Yield_Bond_SPREAD'].std()
                    spread_mean = df_quadrants['High_Yield_Bond_SPREAD'].mean()
                    st.caption(f"**HY Spread:** Moyenne = {spread_mean:.2f}% | Ecart-type = {spread_std:.2f}%")

                with col_i:
                    be_std = df_quadrants['BREAKEVEN_10Y'].std()
                    be_mean = df_quadrants['BREAKEVEN_10Y'].mean()
                    st.caption(f"**Breakeven:** Moyenne = {be_mean:.2f}% | Ecart-type = {be_std:.2f}%")

            # Fallback to Initial Claims / CPI if new targets missing (Legacy)
            elif 'INITIAL_CLAIMS_YOY' in df_quadrants.columns and 'CPI_YOY' in df_quadrants.columns:
                df_recent = df_quadrants[df_quadrants['date'] >= (pd.Timestamp.now() - pd.Timedelta(days=20 * 365))]

                fig_targets = go.Figure()

                # Growth: Initial Claims (Inverted logic for display? No, just plot raw but maybe invert axis)
                # Lower Claims = Growth. Let's plot it as is but mention it.
                fig_targets.add_trace(go.Scatter(
                    x=df_recent['date'], y=df_recent['INITIAL_CLAIMS_YOY'],
                    mode='lines', name='Growth Proxy (Initial Claims YoY %)',
                    line=dict(color='green', width=2)
                ))

                fig_targets.add_trace(go.Scatter(
                    x=df_recent['date'], y=df_recent['CPI_YOY'],
                    mode='lines', name='Inflation Target (CPI YoY %)',
                    line=dict(color='orange', width=2)
                ))

                fig_targets.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

                fig_targets.update_layout(
                    height=350, xaxis_title="Date", yaxis_title="Year-over-Year Change (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                # Invert Y axis? No, that might be confusing if CPI is on same chart.
                # Better to just label it clearly.
                
                st.plotly_chart(fig_targets, use_container_width=True)

                col_g, col_i = st.columns(2)
                with col_g:
                    claims_std = df_quadrants['INITIAL_CLAIMS_YOY'].std()
                    claims_mean = df_quadrants['INITIAL_CLAIMS_YOY'].mean()
                    st.caption(f"**Claims (Growth Inv):** Moyenne = {claims_mean:.2f}% | Ecart-type = {claims_std:.2f}%")

                with col_i:
                    inflation_std = df_quadrants['CPI_YOY'].std()
                    inflation_mean = df_quadrants['CPI_YOY'].mean()
                    st.caption(f"**Inflation:** Moyenne = {inflation_mean:.2f}% | Ecart-type = {inflation_std:.2f}%")
            
            else:
                st.info("Targets (Claims/USPHCI or CPI) non disponibles dans les donnees.")
        else:
            st.info("Donnees quadrants non disponibles.")

        st.divider()

        # ========================================
        # GRIDSEARCH CV RESULTS
        # ========================================
        st.subheader("GridSearchCV - Resultats Optimisation")

        gs_growth = metrics.get('gridsearch_growth_results')
        gs_inflation = metrics.get('gridsearch_inflation_results')

        if gs_growth and gs_inflation:
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                best_g = metrics.get('rf_params_growth', {})
                cv_g = metrics.get('gridsearch_growth_cv_score', 0)
                cv_label = 'AUC' if is_classifier else 'R2'
                st.metric(f"Best Risk CV {cv_label}", f"{cv_g:.3f}")
                st.caption(f"Params: {best_g}")
            with col_bp2:
                best_i = metrics.get('rf_params_inflation', {})
                cv_i = metrics.get('gridsearch_inflation_cv_score', 0)
                st.metric(f"Best Rates CV {cv_label}", f"{cv_i:.3f}")
                st.caption(f"Params: {best_i}")

            # Top 10 chart
            col_gs1, col_gs2 = st.columns(2)
            gs_label = 'AUC' if is_classifier else 'R2'

            with col_gs1:
                st.caption("**Top 10 - Risk Model**")
                labels = [f"#{r['rank']}" for r in gs_growth]
                train_scores = [r['mean_train_score'] for r in gs_growth]
                test_scores = [r['mean_test_score'] for r in gs_growth]
                hover_texts = [str(r['params']) for r in gs_growth]

                fig_gs_g = go.Figure()
                fig_gs_g.add_trace(go.Bar(x=labels, y=train_scores, name=f'Train {gs_label}', marker_color='rgba(0,200,0,0.6)', hovertext=hover_texts))
                fig_gs_g.add_trace(go.Bar(x=labels, y=test_scores, name=f'Test {gs_label} (CV)', marker_color='rgba(0,200,0,1)', hovertext=hover_texts))
                fig_gs_g.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_gs_g.update_layout(height=300, barmode='group', yaxis_title=f'{gs_label} Score', xaxis_title='Rank')
                st.plotly_chart(fig_gs_g, use_container_width=True)

            with col_gs2:
                st.caption("**Top 10 - Rates Model**")
                labels = [f"#{r['rank']}" for r in gs_inflation]
                train_scores = [r['mean_train_score'] for r in gs_inflation]
                test_scores = [r['mean_test_score'] for r in gs_inflation]
                hover_texts = [str(r['params']) for r in gs_inflation]

                fig_gs_i = go.Figure()
                fig_gs_i.add_trace(go.Bar(x=labels, y=train_scores, name=f'Train {gs_label}', marker_color='rgba(255,165,0,0.6)', hovertext=hover_texts))
                fig_gs_i.add_trace(go.Bar(x=labels, y=test_scores, name=f'Test {gs_label} (CV)', marker_color='rgba(255,165,0,1)', hovertext=hover_texts))
                fig_gs_i.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_gs_i.update_layout(height=300, barmode='group', yaxis_title=f'{gs_label} Score', xaxis_title='Rank')
                st.plotly_chart(fig_gs_i, use_container_width=True)

            # Overfitting indicator
            if gs_growth[0]['mean_train_score'] - gs_growth[0]['mean_test_score'] > 0.3:
                st.warning("**Growth:** Ecart Train/Test > 0.3 -> risque d'overfitting meme avec les meilleurs params")
            if gs_inflation[0]['mean_train_score'] - gs_inflation[0]['mean_test_score'] > 0.3:
                st.warning("**Inflation:** Ecart Train/Test > 0.3 -> risque d'overfitting meme avec les meilleurs params")
        else:
            st.info("Resultats GridSearchCV non disponibles. Relancez compute_quadrants.py pour generer.")

        st.divider()

        # ========================================
        # CONFUSION MATRIX AND FEATURE IMPORTANCE
        # ========================================
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix (OOS)")
            st.caption("Lignes = Quadrant Reel | Colonnes = Quadrant Predit")

            cm = metrics.get('confusion_matrix_out_of_sample', metrics.get('confusion_matrix', [[0] * 4] * 4))
            cm_df = pd.DataFrame(cm,
                                 index=['Q1 Real', 'Q2 Real', 'Q3 Real', 'Q4 Real'],
                                 columns=['Q1 Pred', 'Q2 Pred', 'Q3 Pred', 'Q4 Pred'])

            fig_cm = px.imshow(cm_df,
                               text_auto=True,
                               color_continuous_scale='Blues',
                               aspect='auto')
            fig_cm.update_layout(height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

            diagonal_sum = sum(cm[i][i] for i in range(4))
            total = sum(sum(row) for row in cm)
            if total > 0:
                st.info(f"{diagonal_sum}/{total} classifications correctes ({diagonal_sum / total:.1%})")

        with col_right:
            st.subheader("Feature Importance")

            # Growth Model
            st.markdown("**Risk Model (Top 5)**")
            fi_growth = metrics.get('feature_importance_growth', {})
            fi_growth_sorted = dict(sorted(fi_growth.items(), key=lambda x: -x[1])[:5])

            fig_fi_g = go.Figure(go.Bar(
                x=list(fi_growth_sorted.values()),
                y=list(fi_growth_sorted.keys()),
                orientation='h', marker_color='green'
            ))
            fig_fi_g.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi_g, use_container_width=True)

            # Inflation Model
            st.markdown("**Rates Model (Top 5)**")
            fi_inflation = metrics.get('feature_importance_inflation', {})
            fi_inflation_sorted = dict(sorted(fi_inflation.items(), key=lambda x: -x[1])[:5])

            fig_fi_i = go.Figure(go.Bar(
                x=list(fi_inflation_sorted.values()),
                y=list(fi_inflation_sorted.keys()),
                orientation='h', marker_color='orange'
            ))
            fig_fi_i.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fi_i, use_container_width=True)

        st.divider()

        # Model Configuration
        with st.expander("Configuration du Modele"):
            model_type_label = "Random Forest Classifier" if is_classifier else "Random Forest Regressor"
            rf_params_g = metrics.get('rf_params_growth', metrics.get('rf_params', {}))
            timestamp = metrics.get('timestamp', 'N/A')
            training_samples = metrics.get('training_samples', 0)
            rolling_window = metrics.get('rolling_median_window', 'N/A')

            st.markdown(f"""
            | Parametre | Valeur |
            |-----------|--------|
            | **Algorithme** | {model_type_label} |
            | **n_estimators** | {rf_params_g.get('n_estimators', 'N/A')} |
            | **max_depth** | {rf_params_g.get('max_depth', 'N/A')} |
            | **min_samples_leaf** | {rf_params_g.get('min_samples_leaf', 'N/A')} |
            | **Rolling Median Window** | {rolling_window} jours |
            | **Training Samples** | {training_samples} mois |
            | **Validation** | Walk-Forward (2008+) |
            | **Derniere MAJ** | {timestamp[:19] if timestamp != 'N/A' else 'N/A'} |
            """)
    else:
        st.warning("Donnees ML non disponibles. Lancez le DAG pour generer les metriques.")
        st.code("airflow dags trigger dag_us_macro", language="bash")
