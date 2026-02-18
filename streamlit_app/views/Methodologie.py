"""
Page 4: Methodologie & Tech
Technical documentation: indicators, architecture, allocations.
"""

import streamlit as st


def render(data):
    st.header("Sous le capot")

    with st.expander("1. Definition des Indicateurs (Feature Engineering)", expanded=True):
        st.markdown("""
        Pour definir la position economique, j'utilise des donnees de la FED et Yahoo Finance :
        
        **Axe Croissance (Growth):**
        - Housing Permits, Industrial Production, Consumer Sentiment, Copper, 10-2Y Yield Curve
        - Negatif : Initial Claims, High Yield Spread, VIX, Real Rates
        
        **Axe Inflation:**
        - CPI, **10Y Breakeven Inflation Rate** (T10YIE), WTI Oil Prices
        - Negatif : US Dollar Index (Dollar fort = Deflationniste)
        """)
        st.latex(r'''
        Score_{position} = \frac{X_t - \mu_{expanding}}{\sigma_{expanding}}
        ''')
        st.caption("J'utilise des Z-Scores rolling pour normaliser les donnees et detecter les changements de regime.")

    with st.expander("2. Architecture Data (Airflow)"):
        st.markdown("""
        L'automatisation est geree par **Apache Airflow**.
        1.  **Task `fetch_data`** : Recupere les donnees brutes (FRED, Yahoo Finance).
        2.  **Task `prepare_indicators`** : Merge et nettoie les indicateurs.
        3.  **Spark Job `train_model`** : Entraine les classifieurs (GridSearchCV + Walk-Forward).
        4.  **Spark Job `compute_quadrants`** : Infere les probabilites et assigne les quadrants.
        5.  **Spark Job `backtest_strategy`** : Simule la strategie avec couts de transaction.
        """)
        st.code("""
def assign_quadrant(prob_growth, prob_inflation):
    if prob_growth > 0.5 and prob_inflation < 0.5:  return 1  # Goldilocks
    if prob_growth > 0.5 and prob_inflation >= 0.5: return 2  # Reflation
    if prob_growth <= 0.5 and prob_inflation >= 0.5: return 3  # Stagflation
    return 4  # Deflation
        """, language="python")

    with st.expander("3. Allocations par Quadrant"):
        st.markdown("""
        | Quadrant | SP500 | NASDAQ | SmallCAP | GOLD | COMMODITIES | TREASURY |
        |----------|-------|--------|----------|------|-------------|----------|
        | Q1 Growth | 30% | 40% | 30% | 0% | 0% | 0% |
        | Q2 Inflation | 40% | 10% | 0% | 30% | 20% | 0% |
        | Q3 Stagflation | 0% | 0% | 0% | 60% | 20% | 20% |
        | Q4 Deflation | 0% | 0% | 0% | 40% | 0% | 60% |
        """)

    with st.expander("4. Modele ML (Binary Classification)"):
        st.markdown("""
        **Approche :** Classification binaire avec Random Forest.
        
        **Targets dynamiques :**
        - `TARGET_GROWTH = 1` si USPHCI_YoY > Mediane Glissante (5 ans)
        - `TARGET_INFLATION = 1` si CPI_YoY > Mediane Glissante (5 ans)
        
        **Prediction :**
        - `predict_proba()` -> probabilite P(croissance haute) et P(inflation haute)
        - Lissage EMA (span=5) pour reduire le bruit
        - Quadrant = combinaison des probabilites (seuil 0.5)
        
        **Validation :**
        - Walk-Forward annuel (train sur le passe, test sur l'annee suivante)
        - Metriques : Accuracy, Precision, Recall, AUC-ROC
        """)
