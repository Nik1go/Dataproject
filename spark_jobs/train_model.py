"""
train_model.py - ML Training Pipeline (Binary Classification)

Trains Random Forest CLASSIFIERS for Growth and Inflation regimes.
Uses rolling median as dynamic threshold for binary targets.

Outputs:
    - ml_pipeline.pkl  (classifiers + scaler + feature_cols + rolling medians)
    - ml_metrics.json   (Accuracy, Precision, Recall, AUC-ROC for Streamlit)

Usage:
    cd ~/airflow
    source airflow_venv/bin/activate
    python spark_jobs/train_model.py \
        data/US/output_dag/combined_indicators.csv \
        data/US/output_dag
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import pickle


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# ============================================================
# CONFIGURATION
# ============================================================

# Publication Lags (in TRADING DAYS)
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
    'INFLATION': 30,
    'USPHCI': 60,
    'Real_Gross_Domestic_Product': 60,
}

# Rolling Median Window for Dynamic Threshold (5 years = 1260 trading days)
ROLLING_MEDIAN_WINDOW = 1260

# GridSearchCV Hyperparameter Grid (Classifier)
# GridSearchCV Hyperparameter Grid (Classification)
PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [10, 30, 50],
    'max_features': ['sqrt', 0.3]
}

# Feature columns
# Feature columns
FEATURE_COLS_CANDIDATES = [
    'COPPER',
    '10-2Year_Treasury_Yield_Bond', 'INITIAL_CLAIMS', 'High_Yield_Bond_SPREAD',
    'VIX', 'BREAKEVEN_10Y', 'WTI_CRUDE_OIL', 'US_DOLLAR_INDEX', 'REAL_RATES',
    'NET_LIQUIDITY',
    # New Fundamental/Macro Features
    'CONSUMER_SENTIMENT', 'HOUSING_PERMITS', 'IND_PRODUCTION', 'INFLATION_YOY',
    # New Features (Dynamic Momentum & Volatility - Pruned)
    'COPPER_MOM_1M', 'COPPER_VOL_1M',
    'COPPER_MOM_3M', 'COPPER_VOL_3M',
    'WTI_CRUDE_OIL_MOM_1M', 'WTI_CRUDE_OIL_VOL_1M',
    'WTI_CRUDE_OIL_MOM_3M', 'WTI_CRUDE_OIL_VOL_3M',
    'US_DOLLAR_INDEX_MOM_1M', 'US_DOLLAR_INDEX_VOL_1M',
    'US_DOLLAR_INDEX_MOM_3M', 'US_DOLLAR_INDEX_VOL_3M',
    'High_Yield_Bond_SPREAD_MOM_1M', 'High_Yield_Bond_SPREAD_VOL_1M',
    'High_Yield_Bond_SPREAD_MOM_3M', 'High_Yield_Bond_SPREAD_VOL_3M',
    '10-2Year_Treasury_Yield_Bond_MOM_1M', '10-2Year_Treasury_Yield_Bond_VOL_1M',
    '10-2Year_Treasury_Yield_Bond_MOM_3M', '10-2Year_Treasury_Yield_Bond_VOL_3M',
    'VIX_MOM_1M', 'VIX_VOL_1M',
    'VIX_VOL_3M', # Removed VIX_MOM_3M (Pump & Dump)
    'NET_LIQUIDITY_MOM_3M', 'NET_LIQUIDITY_MOM_6M',
    'REAL_RATES_MOM_3M', 'REAL_RATES_MOM_6M',
    'CONSUMER_SENTIMENT_MOM_3M', 'CONSUMER_SENTIMENT_MOM_6M',
    'HOUSING_PERMITS_MOM_3M', 'HOUSING_PERMITS_MOM_6M',
    'IND_PRODUCTION_MOM_3M', 'IND_PRODUCTION_MOM_6M',
    'INFLATION_YOY_MOM_3M', 'INFLATION_YOY_MOM_6M'
]


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


def create_binary_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification targets using Trend-Following Logic (SMA 3M vs 6M).
    
    TARGET_RISK_CLASS (old Growth) = 1 if Spread SMA_3M < SMA_6M (Falling Spreads = Risk On)
    TARGET_INFLATION_CLASS = 1 if Breakeven SMA_3M > SMA_6M (Rising Breakevens = Reflation)
    """
    # ---------------------------------------------------------
    # 1. RISK REGIME (High Yield Bond Spread) - Replaces Growth
    # ---------------------------------------------------------
    if 'High_Yield_Bond_SPREAD' in df.columns:
        # Calculate SMAs (1M = 21 days, 3M = 63 days approx) - FASTER REACTION
        df['SPREAD_SMA_1M'] = df['High_Yield_Bond_SPREAD'].rolling(window=21).mean()
        df['SPREAD_SMA_3M'] = df['High_Yield_Bond_SPREAD'].rolling(window=63).mean()
        
        # Risk On (1) if Spreads are FALLING (1M < 3M)
        # Risk Off (0) if Spreads are RISING (1M > 3M)
        df['TARGET_RISK_CLASS'] = (df['SPREAD_SMA_1M'] < df['SPREAD_SMA_3M']).astype(int)
    else:
        raise ValueError("High_Yield_Bond_SPREAD missing for Risk Target generation")

    # ---------------------------------------------------------
    # 2. INFLATION REGIME (10Y Breakeven)
    # ---------------------------------------------------------
    if 'BREAKEVEN_10Y' in df.columns:
        # Calculate SMAs (1M = 21 days, 3M = 63 days)
        df['BE_SMA_1M'] = df['BREAKEVEN_10Y'].rolling(window=21).mean()
        df['BE_SMA_3M'] = df['BREAKEVEN_10Y'].rolling(window=63).mean()
        
        # Reflation (1) if Breakevens are RISING (1M > 3M)
        # Disinflation (0) if Breakevens are FALLING (1M < 3M)
        df['TARGET_INFLATION_CLASS'] = (df['BE_SMA_1M'] > df['BE_SMA_3M']).astype(int)
    else:
        raise ValueError("BREAKEVEN_10Y missing for Inflation Target generation")
    
    # Drop NaNs created by rolling windows to avoid training on garbage
    df = df.dropna(subset=['SPREAD_SMA_3M', 'BE_SMA_3M'])
    
    return df


def walk_forward_classification(df_features: pd.DataFrame, df_targets: pd.DataFrame,
                                 feature_cols: list, target_col: str,
                                 start_year: int = 2010, min_train_years: int = 3):
    """
    Walk-Forward Validation for Binary Classification.
    
    Returns per-year and overall Accuracy, AUC-ROC, and predictions.
    """
    results = {
        'per_year_accuracy': {},
        'per_year_auc': {},
        'per_year_samples': {},
        'overall_accuracy': None,
        'overall_auc': None,
        'oos_probabilities': [],
        'oos_predictions': [],
        'oos_true': [],
        'oos_dates': []
    }
    
    all_years = sorted(df_features.index.year.unique())
    test_years = [y for y in all_years if y >= start_year]
    
    all_probs = []
    all_preds = []
    all_true = []
    all_dates = []
    
    for test_year in test_years:
        train_idx = df_features.index[df_features.index.year < test_year]
        test_idx = df_features.index[df_features.index.year == test_year]
        
        if len(train_idx) < min_train_years * 12:
            continue
        
        X_train = df_features.loc[train_idx, feature_cols]
        y_train = df_targets.loc[train_idx, target_col]
        X_test = df_features.loc[test_idx, feature_cols]
        y_test = df_targets.loc[test_idx, target_col]
        
        # Drop NaN values and handle Infinity
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        if len(X_train) < 12 or len(X_test) < 1:
            continue
        
        # Need both classes in training data to learn
        if len(y_train.unique()) < 2:
            continue
        
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train_sc, y_train)
        
        probs = model.predict_proba(X_test_sc)[:, 1]  # Probability of class 1
        preds = (probs >= 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5  # Default if only one class in test set
        
        results['per_year_accuracy'][test_year] = acc
        results['per_year_auc'][test_year] = auc
        results['per_year_samples'][test_year] = len(y_test)
        
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(y_test.values)
        all_dates.extend(y_test.index.tolist())
    
    if len(all_preds) > 0:
        results['overall_accuracy'] = accuracy_score(all_true, all_preds)
        try:
            results['overall_auc'] = roc_auc_score(all_true, all_probs)
        except ValueError:
            results['overall_auc'] = 0.5
        results['oos_probabilities'] = [float(p) for p in all_probs]
        results['oos_predictions'] = [int(p) for p in all_preds]
        results['oos_true'] = [int(t) for t in all_true]
        results['oos_dates'] = all_dates
    
    return results


# ============================================================
# MAIN LOGIC
# ============================================================

def main(indicators_path: str, output_dir: str):
    print("=" * 60)
    print("ML TRAINING PIPELINE (Binary Classification)")
    print(f"Rolling Median Window: {ROLLING_MEDIAN_WINDOW} days")
    print("=" * 60)
    
    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("\n[1/6] Loading data...")
    
    if indicators_path.endswith('.csv'):
        df = pd.read_csv(indicators_path, parse_dates=['date'])
    elif indicators_path.endswith('.parquet'):
        df = pd.read_parquet(indicators_path)
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = pd.read_csv(indicators_path, parse_dates=['date'])
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    print(f"   Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    
    # ========================================
    # 2. FORWARD FILL + LAGS
    # ========================================
    print("\n[2/6] Forward filling + applying publication lags...")
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
    
    # ========================================
    # 3. CREATE TARGET VARIABLES
    # ========================================
    print("\n[3/6] Creating binary classification targets ...")
    
    # Create binary targets with SMA Trend logic
    df = create_binary_targets(df)
    
    growth_balance = df['TARGET_RISK_CLASS'].value_counts(normalize=True)
    inflation_balance = df['TARGET_INFLATION_CLASS'].value_counts(normalize=True)
    print(f"   Growth class balance: {growth_balance.to_dict()}")
    print(f"   Inflation class balance: {inflation_balance.to_dict()}")
    
    # ========================================
    # 4. PREPARE FEATURES
    # ========================================
    print("\n[4/6] Preparing features...")
    
    feature_cols = [c for c in FEATURE_COLS_CANDIDATES if c in df_lagged.columns]
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    
    # Weekly resampling for training (approx 800+ samples)
    df_monthly = df_lagged[feature_cols].resample('W-FRI').last()
    targets_monthly = df[['TARGET_RISK_CLASS', 'TARGET_INFLATION_CLASS']].resample('W-FRI').last()
    
    aligned = df_monthly.join(targets_monthly, how='inner').dropna()
    
    if len(aligned) < 50:
        raise ValueError(f"Not enough data: {len(aligned)} weeks (need 50+)")
    
    X_train = aligned[feature_cols]
    y_growth = aligned['TARGET_RISK_CLASS'].astype(int)
    y_inflation = aligned['TARGET_INFLATION_CLASS'].astype(int)
    
    print(f"   Training samples: {len(X_train)} weeks")
    print(f"   Growth class 1 ratio: {y_growth.mean():.1%}")
    print(f"   Inflation class 1 ratio: {y_inflation.mean():.1%}")
    
    print(f"   Growth class 1 ratio: {y_growth.mean():.1%}")
    print(f"   Inflation class 1 ratio: {y_inflation.mean():.1%}")
    
    # Handle Infinity/NaN before scaling
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    print(f"   Growth class 1 ratio: {y_growth.mean():.1%}")
    print(f"   Inflation class 1 ratio: {y_inflation.mean():.1%}")
    
    # Handle Infinity/NaN before scaling
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    
    # ========================================
    # 5. GRIDSEARCH CV (Classification)
    # ========================================
    print("\n[5/6] Running GridSearchCV (Classification)...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Split features by model type (Specialization)
    # RISK MODEL (Old Growth): Fast signals only
    # EXCLUDE HY SPREAD (Self-leakage) but INCLUDE BREAKEVENS (Cross-confirmation)
    feature_cols_risk = [
        c for c in feature_cols 
        if 'MOM_6M' not in c and 'MOM_12M' not in c  # Keep existing fast signal logic
        and 'High_Yield_Bond_SPREAD' not in c        # Remove Self-Leakage
    ]
    
    # INFLATION MODEL: All features (Hybrid)
    # EXCLUDE BREAKEVENS (Self-leakage) but INCLUDE HY SPREAD (Cross-confirmation)
    feature_cols_inflation = [
        c for c in feature_cols 
        if 'BREAKEVEN_10Y' not in c                  # Remove Self-Leakage
    ] 
    
    # Risk Classifier
    print("   Running GridSearch Risk Classifier...")
    grid_growth = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        return_train_score=True
    )
    grid_growth.fit(X_train_scaled_df[feature_cols_risk], y_growth)
    model_growth = grid_growth.best_estimator_
    print(f"   Best Growth Params: {grid_growth.best_params_}")
    print(f"   Best Growth CV AUC: {grid_growth.best_score_:.3f}")
    
    # Inflation Classifier
    print("   Running GridSearch Inflation Classifier...")
    grid_inflation = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
        param_grid=PARAM_GRID,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        return_train_score=True
    )
    grid_inflation.fit(X_train_scaled_df[feature_cols_inflation], y_inflation)
    model_inflation = grid_inflation.best_estimator_
    print(f"   Best Inflation Params: {grid_inflation.best_params_}")
    print(f"   Best Inflation CV AUC: {grid_inflation.best_score_:.3f}")
    
    # ========================================
    # 6. EVALUATION (Walk-Forward OOS)
    # ========================================
    print("\n[6/6] Walk-Forward Validation (Classification)...")
    
    # Walk-Forward for Risk (Growth variable reused as Risk for compatibility)
    wf_growth = walk_forward_classification(
        df_features=aligned[feature_cols_risk],
        df_targets=aligned,
        feature_cols=feature_cols_risk,
        target_col='TARGET_RISK_CLASS',
        start_year=2005,
        min_train_years=3
    )
    
    # Walk-Forward for Inflation
    wf_inflation = walk_forward_classification(
        df_features=aligned[feature_cols_inflation],
        df_targets=aligned,
        feature_cols=feature_cols_inflation,
        target_col='TARGET_INFLATION_CLASS',
        start_year=2005,
        min_train_years=3
    )
    
    # OOS Results
    acc_growth_oos = wf_growth['overall_accuracy'] or 0.0
    auc_growth_oos = wf_growth['overall_auc'] or 0.5
    acc_inflation_oos = wf_inflation['overall_accuracy'] or 0.0
    auc_inflation_oos = wf_inflation['overall_auc'] or 0.5
    
    print(f"\n   OOS Metrics (Walk-Forward):")
    print(f"      Growth     - Accuracy: {acc_growth_oos:.1%} | AUC: {auc_growth_oos:.3f}")
    print(f"      Inflation  - Accuracy: {acc_inflation_oos:.1%} | AUC: {auc_inflation_oos:.3f}")
    

    
    # OOS Classification Reports
    if len(wf_growth['oos_true']) > 0:
        prec_growth_oos = precision_score(wf_growth['oos_true'], wf_growth['oos_predictions'], zero_division=0)
        rec_growth_oos = recall_score(wf_growth['oos_true'], wf_growth['oos_predictions'], zero_division=0)
    else:
        prec_growth_oos = rec_growth_oos = 0.0
    
    if len(wf_inflation['oos_true']) > 0:
        prec_inflation_oos = precision_score(wf_inflation['oos_true'], wf_inflation['oos_predictions'], zero_division=0)
        rec_inflation_oos = recall_score(wf_inflation['oos_true'], wf_inflation['oos_predictions'], zero_division=0)
    else:
        prec_inflation_oos = rec_inflation_oos = 0.0
    
    # Quadrant Accuracy (OOS)
    if len(wf_growth['oos_probabilities']) > 0 and len(wf_inflation['oos_probabilities']) > 0:
        oos_growth_df = pd.DataFrame({
            'date': wf_growth['oos_dates'],
            'PROB_GROWTH': wf_growth['oos_probabilities'],
            'TRUE_GROWTH': wf_growth['oos_true']
        }).set_index('date')
        
        oos_inflation_df = pd.DataFrame({
            'date': wf_inflation['oos_dates'],
            'PROB_INFLATION': wf_inflation['oos_probabilities'],
            'TRUE_INFLATION': wf_inflation['oos_true']
        }).set_index('date')
        
        oos_combined = oos_growth_df.join(oos_inflation_df, how='inner').dropna()
        
        if len(oos_combined) > 0:
            # Predicted quadrants from probabilities
            oos_combined['PRED_Q'] = np.where(
                (oos_combined['PROB_GROWTH'] > 0.5) & (oos_combined['PROB_INFLATION'] < 0.5), 1,
                np.where(
                    (oos_combined['PROB_GROWTH'] > 0.5) & (oos_combined['PROB_INFLATION'] >= 0.5), 2,
                    np.where(
                        (oos_combined['PROB_GROWTH'] <= 0.5) & (oos_combined['PROB_INFLATION'] >= 0.5), 3, 4
                    )
                )
            )
            # True quadrants from binary labels  
            oos_combined['TRUE_Q'] = np.where(
                (oos_combined['TRUE_GROWTH'] == 1) & (oos_combined['TRUE_INFLATION'] == 0), 1,
                np.where(
                    (oos_combined['TRUE_GROWTH'] == 1) & (oos_combined['TRUE_INFLATION'] == 1), 2,
                    np.where(
                        (oos_combined['TRUE_GROWTH'] == 0) & (oos_combined['TRUE_INFLATION'] == 1), 3, 4
                    )
                )
            )
            
            quadrant_acc_oos = accuracy_score(oos_combined['TRUE_Q'], oos_combined['PRED_Q'])
            cm_oos = confusion_matrix(oos_combined['TRUE_Q'], oos_combined['PRED_Q'], labels=[1, 2, 3, 4])
        else:
            quadrant_acc_oos = 0.0
            cm_oos = [[0]*4]*4
    else:
        quadrant_acc_oos = 0.0
        cm_oos = [[0]*4]*4
    
    print(f"\n   Quadrant Accuracy (OOS): {quadrant_acc_oos:.1%}")
    
    # Feature Importance
    importance_growth = dict(zip(feature_cols_risk, model_growth.feature_importances_))
    importance_inflation = dict(zip(feature_cols_inflation, model_inflation.feature_importances_))
    
    print(f"\n   Feature Importance (Growth - Top 5):")
    for feat, imp in sorted(importance_growth.items(), key=lambda x: -x[1])[:5]:
        print(f"      {feat}: {imp:.3f}")
    
    # ========================================
    # SAVE PIPELINE (single pkl)
    # ========================================
    os.makedirs(output_dir, exist_ok=True)
    
    pipeline = {
        'model_growth': model_growth,
        'model_inflation': model_inflation,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_cols_growth': feature_cols_risk,
        'feature_cols_inflation': feature_cols_inflation,
        'model_type': 'classifier',
        'rolling_median_window': ROLLING_MEDIAN_WINDOW
    }
    
    pipeline_path = os.path.join(output_dir, 'ml_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved -> {pipeline_path}")
    
    # ========================================
    # SAVE METRICS (JSON for Streamlit)
    # ========================================
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'binary_classification',
        'validation_type': 'walk_forward',
        'rolling_median_window': ROLLING_MEDIAN_WINDOW,
        

        
        # Out-of-Sample Metrics (Walk-Forward)
        'accuracy_growth_out_of_sample': float(acc_growth_oos),
        'accuracy_inflation_out_of_sample': float(acc_inflation_oos),
        'auc_growth_out_of_sample': float(auc_growth_oos),
        'auc_inflation_out_of_sample': float(auc_inflation_oos),
        'precision_growth_out_of_sample': float(prec_growth_oos),
        'precision_inflation_out_of_sample': float(prec_inflation_oos),
        'recall_growth_out_of_sample': float(rec_growth_oos),
        'recall_inflation_out_of_sample': float(rec_inflation_oos),
        'accuracy_out_of_sample': float(quadrant_acc_oos),
        'confusion_matrix_out_of_sample': [[int(x) for x in row] for row in cm_oos],
        
        # Per-Year Metrics
        'walk_forward_growth_per_year_accuracy': {str(k): float(v) for k, v in wf_growth['per_year_accuracy'].items()},
        'walk_forward_growth_per_year_auc': {str(k): float(v) for k, v in wf_growth['per_year_auc'].items()},
        'walk_forward_inflation_per_year_accuracy': {str(k): float(v) for k, v in wf_inflation['per_year_accuracy'].items()},
        'walk_forward_inflation_per_year_auc': {str(k): float(v) for k, v in wf_inflation['per_year_auc'].items()},
        'walk_forward_samples_per_year': {str(k): int(float(v)) for k, v in wf_growth['per_year_samples'].items()},
        
        # Legacy fields (backward compatibility with Streamlit)
        'accuracy_score': float(quadrant_acc_oos),
        'confusion_matrix': [[int(x) for x in row] for row in cm_oos],
        
        # Model Config
        'feature_importance_growth': {k: float(v) for k, v in importance_growth.items()},
        'feature_importance_inflation': {k: float(v) for k, v in importance_inflation.items()},
        'training_samples': int(len(X_train)),
        'rf_params_growth': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in grid_growth.best_params_.items()},
        'rf_params_inflation': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in grid_inflation.best_params_.items()},
        'gridsearch_growth_cv_score': float(grid_growth.best_score_),
        'gridsearch_inflation_cv_score': float(grid_inflation.best_score_),
        'param_grid': {k: [str(x) if not isinstance(x, (int, float)) else x for x in v] for k, v in PARAM_GRID.items()},
        
        # Top 10 GridSearch results
        'gridsearch_growth_results': sorted([
            {
                'params': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in r['params'].items()},
                'mean_test_score': float(r['mean_test_score']),
                'mean_train_score': float(r['mean_train_score']),
                'rank': int(r['rank_test_score'])
            }
            for r in [dict(zip(grid_growth.cv_results_.keys(), v)) for v in zip(*grid_growth.cv_results_.values())]
        ], key=lambda x: x['rank'])[:10],
        
        'gridsearch_inflation_results': sorted([
            {
                'params': {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in r['params'].items()},
                'mean_test_score': float(r['mean_test_score']),
                'mean_train_score': float(r['mean_train_score']),
                'rank': int(r['rank_test_score'])
            }
            for r in [dict(zip(grid_inflation.cv_results_.keys(), v)) for v in zip(*grid_inflation.cv_results_.values())]
        ], key=lambda x: x['rank'])[:10],
        
        'feature_cols': feature_cols,
        'feature_cols_growth': feature_cols_risk,
        'feature_cols_inflation': feature_cols_inflation
    }
    
    metrics_path = os.path.join(output_dir, 'ml_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")
    
    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <indicators.csv> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
