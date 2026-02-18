"""
compute_quadrants.py - Economic Quadrant Classification (Probability-Based)

Loads pre-trained ML classifiers from ml_pipeline.pkl, runs daily inference
using predict_proba, applies EMA smoothing, and assigns economic quadrants.

Usage:
    cd ~/airflow
    source airflow_venv/bin/activate
    python spark_jobs/compute_quadrants.py \
        data/US/output_dag/combined_indicators.csv \
        data/US/output_dag/ml_pipeline.pkl \
        data/US/output_dag/quadrants.parquet \
        data/US/output_dag/quadrants.csv
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle


# ============================================================
# CONFIGURATION
# ============================================================

# Publication Lags (in TRADING DAYS)
LAGS_TRADING_DAYS = {
    'WTI_CRUDE_OIL': 0,
    'US_DOLLAR_INDEX': 0,
    'VIX': 0,
    'BREAKEVEN_10Y': 0,
    'High_Yield_Bond_SPREAD': 0,
    '10-2Year_Treasury_Yield_Bond': 0,
    'COPPER': 0,
    'TAUX_FED': 0,
    'NET_LIQUIDITY': 0,

    'IND_PRODUCTION': 35,
    'HOUSING_PERMITS': 25,
    'CONSUMER_SENTIMENT': 5,
    'INITIAL_CLAIMS': 5,
    'INFLATION': 30,
    'USPHCI': 60,
    'Real_Gross_Domestic_Product': 60,
}

# EMA Smoothing Span for Probabilities
EMA_SPAN = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_publication_lags(df: pd.DataFrame, lags: dict) -> pd.DataFrame:
    """Apply publication lags to features (shift data forward)."""
    df_lagged = df.copy()
    for col, lag_days in lags.items():
        if col in df_lagged.columns and lag_days > 0:
            df_lagged[col] = df_lagged[col].shift(lag_days)
    return df_lagged


def calculate_yoy_change(series: pd.Series, periods: int = 252) -> pd.Series:
    """Calculate Year-over-Year percentage change."""
    return series.pct_change(periods=periods) * 100


def calculate_momentum(series: pd.Series, months: int) -> pd.Series:
    """Calculate Momentum as percentage change over N months (approx 21 days/month)."""
    return series.pct_change(periods=int(months * 21)) * 100


def calculate_volatility(series: pd.Series, months: int) -> pd.Series:
    """Calculate Volatility as rolling standard deviation of daily returns over N months."""
    return series.pct_change(1).rolling(window=int(months * 21)).std() * 100


# ============================================================
# MAIN LOGIC
# ============================================================

def main(indicators_path: str, pipeline_path: str, output_parquet: str, output_csv: str):
    print("=" * 60)
    print("ECONOMIC QUADRANT CLASSIFICATION (Probability-Based)")
    print(f"EMA Span: {EMA_SPAN} days")
    print("=" * 60)
    
    # ========================================
    # 1. LOAD ML PIPELINE
    # ========================================
    print("\n[1/5] Loading ML pipeline...")
    
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    model_growth = pipeline['model_growth']
    model_inflation = pipeline['model_inflation']
    scaler = pipeline['scaler']
    feature_cols = pipeline['feature_cols']
    model_type = pipeline.get('model_type', 'classifier')
    
    print(f"   Loaded {model_type} models + scaler with {len(feature_cols)} features")
    
    # ========================================
    # 2. LOAD & PREPARE DATA
    # ========================================
    print("\n[2/5] Loading data...")
    
    if indicators_path.endswith('.csv'):
        df = pd.read_csv(indicators_path, parse_dates=['date'])
    elif indicators_path.endswith('.parquet'):
        df = pd.read_parquet(indicators_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = pd.read_csv(indicators_path, parse_dates=['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    df = df.ffill()
    
    df_lagged = apply_publication_lags(df, LAGS_TRADING_DAYS)
    
    # Derived feature: Real Rates
    # Derived feature: Real Rates & Inflation YoY
    if 'INFLATION' in df_lagged.columns:
        df_lagged['INFLATION_YOY'] = calculate_yoy_change(df_lagged['INFLATION'], periods=252)

    if 'TAUX_FED' in df_lagged.columns and 'INFLATION_YOY' in df_lagged.columns:
        df_lagged['REAL_RATES'] = df_lagged['TAUX_FED'] - df_lagged['INFLATION_YOY']

    # Derived Features: Momentum & Volatility
    # Split into Fast (Market) and Slow (Macro/Liquidity) for feature selection
    
    fast_assets = [
        'COPPER', 'WTI_CRUDE_OIL', 'US_DOLLAR_INDEX', 
        'High_Yield_Bond_SPREAD', '10-2Year_Treasury_Yield_Bond', 
        'VIX'
    ]
    
    slow_assets = [
        'NET_LIQUIDITY', 'REAL_RATES', 
        'CONSUMER_SENTIMENT', 'HOUSING_PERMITS', 'IND_PRODUCTION', 'INFLATION_YOY'
    ]
    
    # Fast Assets: Focus on 1M & 3M Momentum & Volatility (Reactive)
    for asset in fast_assets:
        if asset in df_lagged.columns:
            # 1M Momentum (Fastest)
            df_lagged[f'{asset}_MOM_1M'] = calculate_momentum(df_lagged[asset], months=1)
            # 3M Momentum (Confirmation)
            df_lagged[f'{asset}_MOM_3M'] = calculate_momentum(df_lagged[asset], months=3)
            
            # Volatility (Rolling Std of Daily Returns) - 1M & 3M
            df_lagged[f'{asset}_VOL_1M'] = calculate_volatility(df_lagged[asset], months=1)
            df_lagged[f'{asset}_VOL_3M'] = calculate_volatility(df_lagged[asset], months=3)

    # Slow Assets: Focus on 3M & 6M Momentum (Trend)
    for asset in slow_assets:
        if asset in df_lagged.columns:
            df_lagged[f'{asset}_MOM_3M'] = calculate_momentum(df_lagged[asset], months=3)
            df_lagged[f'{asset}_MOM_6M'] = calculate_momentum(df_lagged[asset], months=6)
    
    # Create YoY variables (for export in quadrants.csv) - Updated for new Growth proxy
    df['INITIAL_CLAIMS_YOY'] = calculate_yoy_change(df['INITIAL_CLAIMS'], periods=252)
    df['CPI_YOY'] = calculate_yoy_change(df['INFLATION'], periods=252)
    
    # Keep USPHCI just for backup/reference if needed in CSV, but not used for Targets anymore
    if 'USPHCI' in df.columns:
        df['USPHCI_YOY'] = calculate_yoy_change(df['USPHCI'], periods=252)
    
    print(f"   Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # ========================================
    # 3. DAILY INFERENCE (PROBABILITIES)
    # ========================================
    print("\n[3/5] Running daily inference (predict_proba)...")
    
    X_daily = df_lagged[feature_cols].copy()
    X_daily = X_daily.replace([np.inf, -np.inf], np.nan)
    X_daily = X_daily.ffill().bfill()
    
    # Scale features using the loaded scaler (RobustScaler or StandardScaler)
    X_daily_scaled = scaler.transform(X_daily)
    X_daily_scaled_df = pd.DataFrame(X_daily_scaled, columns=feature_cols, index=X_daily.index)
    
    # Handle Split Features (Growth vs Inflation) if pipeline has them
    feature_cols_growth = pipeline.get('feature_cols_growth', feature_cols)
    feature_cols_inflation = pipeline.get('feature_cols_inflation', feature_cols)
    
    # Get probability of class 1 (high growth / high inflation)
    df['PROB_GROWTH_RAW'] = model_growth.predict_proba(X_daily_scaled_df[feature_cols_growth])[:, 1]
    df['PROB_INFLATION_RAW'] = model_inflation.predict_proba(X_daily_scaled_df[feature_cols_inflation])[:, 1]
    
    # ========================================
    # 4. EMA SMOOTHING + QUADRANT ASSIGNMENT
    # ========================================
    print(f"\n[4/5] Applying EMA smoothing (span={EMA_SPAN}) + Quadrant assignment...")
    
    # EMA smoothing on raw probabilities
    df['PROB_GROWTH_EMA'] = df['PROB_GROWTH_RAW'].ewm(span=EMA_SPAN, min_periods=1).mean()
    df['PROB_INFLATION_EMA'] = df['PROB_INFLATION_RAW'].ewm(span=EMA_SPAN, min_periods=1).mean()
    
    # Quadrant Assignment based on smoothed probabilities
    # PROB_GROWTH is now PROB_RISK_ON
    # PROB_INFLATION is PROB_REFLATION
    conditions = [
        (df['PROB_GROWTH_EMA'] > 0.5) & (df['PROB_INFLATION_EMA'] < 0.5),   # Q1: Goldilocks (Risk On + Disinflation)
        (df['PROB_GROWTH_EMA'] > 0.5) & (df['PROB_INFLATION_EMA'] >= 0.5),  # Q2: Reflation Boom (Risk On + Reflation)
        (df['PROB_GROWTH_EMA'] <= 0.5) & (df['PROB_INFLATION_EMA'] >= 0.5), # Q3: Stagflation (Risk Off + Reflation)
        (df['PROB_GROWTH_EMA'] <= 0.5) & (df['PROB_INFLATION_EMA'] < 0.5)    # Q4: Recession (Risk Off + Disinflation)
    ]
    df['assigned_quadrant'] = np.select(conditions, [1, 2, 3, 4], default=1)
    
    # Score columns for backward compatibility with Streamlit scatter plot
    # Map probabilities from [0,1] to [-2,2] for Z-score-like visualization
    df['MACRO_GROWTH_SCORE'] = (df['PROB_GROWTH_EMA'] - 0.5) * 4
    df['MACRO_INFLATION_SCORE'] = (df['PROB_INFLATION_EMA'] - 0.5) * 4
    
    # Legacy columns
    df['score_Q1'] = df['MACRO_GROWTH_SCORE']
    df['score_Q2'] = df['MACRO_INFLATION_SCORE']
    df['score_Q3'] = 0.0
    df['score_Q4'] = 0.0
    
    # ========================================
    # 4b. TARGET QUADRANT (Ground Truth)
    # ========================================
    # Compute the "perfect" quadrant based on Market Logic (Spread & Breakeven SMAs)
    print(f"\n   Computing target quadrants (Market-Implied Risk & Inflation)...")
    
    # 1. RISK TRUTH (Spread 1M vs 3M)
    if 'High_Yield_Bond_SPREAD' in df.columns:
        # 1M = 21 days, 3M = 63 days
        df['SPREAD_SMA_1M'] = df['High_Yield_Bond_SPREAD'].rolling(window=21).mean()
        df['SPREAD_SMA_3M'] = df['High_Yield_Bond_SPREAD'].rolling(window=63).mean()
        # Risk On (1) if Spreads Falling (1M < 3M)
        target_risk_on = (df['SPREAD_SMA_1M'] < df['SPREAD_SMA_3M']).astype(int)
    else:
        target_risk_on = 0
        
    # 2. INFLATION TRUTH (Breakeven 1M vs 3M)
    if 'BREAKEVEN_10Y' in df.columns:
        # 1M = 21 days, 3M = 63 days
        df['BE_SMA_1M'] = df['BREAKEVEN_10Y'].rolling(window=21).mean()
        df['BE_SMA_3M'] = df['BREAKEVEN_10Y'].rolling(window=63).mean()
        # Reflation (1) if Breakevens Rising (1M > 3M)
        target_reflation = (df['BE_SMA_1M'] > df['BE_SMA_3M']).astype(int)
    else:
        target_reflation = 0

    # Quadrant Mapping (Updated)
    # Q1 (Goldilocks)     : Risk On (1) + Disinflation (0)
    # Q2 (Reflation Boom) : Risk On (1) + Reflation (1)
    # Q3 (Stagflation)    : Risk Off (0) + Reflation (1)
    # Q4 (Recession)      : Risk Off (0) + Disinflation (0)
    target_conditions = [
        (target_risk_on == 1) & (target_reflation == 0),   # Q1
        (target_risk_on == 1) & (target_reflation == 1),   # Q2
        (target_risk_on == 0) & (target_reflation == 1),   # Q3
        (target_risk_on == 0) & (target_reflation == 0)    # Q4
    ]
    df['target_quadrant'] = np.select(target_conditions, [1, 2, 3, 4], default=0)
    
    # ========================================
    # 5. OUTPUT
    # ========================================
    print(f"\n[5/5] Writing output...")
    
    df_out = df.reset_index()
    
    df_out.to_parquet(output_parquet, index=False)
    print(f"   Parquet -> {output_parquet}")
    
    df_out.to_csv(output_csv, index=False)
    print(f"   CSV -> {output_csv}")
    
    # Summary
    latest = df.iloc[-1]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Latest Date: {df.index[-1]}")
    print(f"P(High Growth):    {latest['PROB_GROWTH_EMA']:.1%}")
    print(f"P(High Inflation): {latest['PROB_INFLATION_EMA']:.1%}")
    print(f"Assigned Quadrant: Q{int(latest['assigned_quadrant'])}")
    
    # Quadrant distribution
    q_counts = df['assigned_quadrant'].value_counts().sort_index()
    q_names = {1: 'Goldilocks', 2: 'Reflation Boom', 3: 'Stagflation', 4: 'Recession'}
    for q, count in q_counts.items():
        pct = count / len(df) * 100
        print(f"   Q{q} ({q_names[q]}): {count} days ({pct:.1f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python compute_quadrants.py <indicators.csv> <pipeline.pkl> <output.parquet> <output.csv>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
