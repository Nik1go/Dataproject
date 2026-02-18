
import os
import sys
import pandas as pd
import numpy as np

""" compute_assets_performance.py

Objectif :
Ce script Spark (exécuté via `spark-submit`) analyse la performance des actifs financiers
en fonction des quadrants économiques (croissance/inflation/déflation/récession) détectés auparavant.

Il calcule, pour chaque combinaison (actif, quadrant) :
- le rendement mensuel moyen,
- le max drawdown (perte maximale en période de repli),
- le Sharpe ratio annualisé (mesure du couple rendement/risque ajusté de la volatilité).

Étapes principales :
1. Chargement du fichier des quadrants économiques (`quadrant_file`, CSV ou Parquet).
2. Chargement du fichier des valeurs quotidiennes des actifs (`assets_file`, Parquet).
3. Transformation des données au format "long" pour faciliter les calculs par actif/quadrant.
4. Calculs statistiques :
   - Rendement mensuel par actif/quadrant
   - Drawdown maximum
   - Ratio de Sharpe (annualisé)
5. Sauvegarde du tableau synthétique dans un fichier `.parquet` + `.csv`

Exemple de sortie :
| asset_id | assigned_quadrant | monthly_return | max_drawdown | sharpe_annualized |
|----------|-------------------|----------------|--------------|-------------------|
| SP500    | 2 (Stagflation)   | +3.2%          | -8.7%        | 1.12              |

Usage :
Ce script attend 3 arguments :
```bash
python spark_jobs/compute_assets_performance.py data/US/output_dag/quadrants.parquet data/US/output_dag/Assets_daily.parquet data/US/output_dag/assets_performance_by_quadrant.parquet [quadrant_column]
"""
def compute_performance(df_long, df_quadrant, quadrant_col, output_path):
    """Calcule la performance par quadrant et sauvegarde le résultat."""
    print(f"\n[compute_assets_performance] Calcul pour '{quadrant_col}' -> {output_path}")
    
    if quadrant_col not in df_quadrant.columns:
        print(f"⚠️ Colonne '{quadrant_col}' absente du fichier quadrant. Skip.")
        return

    # Prepare quadrant data
    df_q = df_quadrant[['date', quadrant_col]].copy()
    df_q = df_q.sort_values('date')
    
    # FIX: Shift quadrant by 1 day to avoid look-ahead bias
    # Signal at Close(T) applies to Return(T+1)
    df_q[quadrant_col] = df_q[quadrant_col].shift(1)
    
    df_q = df_q.rename(columns={quadrant_col: 'assigned_quadrant'})
    df_q = df_q.dropna(subset=['assigned_quadrant'])

    # Merge asset returns with quadrants
    df_merged = pd.merge(
        left=df_long,
        right=df_q,
        on='date',
        how='inner'
    )

    #  Calculs de performance par (actif, quadrant) sur base DAILY
    rows = []
    grouped = df_merged.groupby(['asset_id', 'assigned_quadrant'])

    for (asset, quadrant), sub in grouped:
        sub = sub.sort_values('date')
        daily_ret = sub['ret'].dropna()
        
        if len(daily_ret) < 2:
            continue

        # Stats
        mean_daily_ret = daily_ret.mean()
        std_daily_ret = daily_ret.std()
        
        annual_return = mean_daily_ret * 252
        sharpe_annual = (mean_daily_ret / std_daily_ret) * np.sqrt(252) if std_daily_ret > 0 else np.nan

        # Max Drawdown
        cumprod = (1 + daily_ret).cumprod()
        rolling_max = cumprod.cummax()
        drawdown = (cumprod - rolling_max) / rolling_max
        max_dd = drawdown.min()

        rows.append({
            'asset': asset,
            'quadrant': int(quadrant),
            'annual_return': annual_return,
            'sharpe': sharpe_annual,
            'max_drawdown': -max_dd,
            'nb_days': len(daily_ret)
        })

    df_summary = pd.DataFrame(rows)
    
    # Ensure directory exists
    parent_out = os.path.dirname(output_path)
    if parent_out and not os.path.isdir(parent_out):
        os.makedirs(parent_out, exist_ok=True)

    df_summary.to_parquet(output_path, index=False)
    print(f"   Shape: {df_summary.shape}")
    print(f"   Parquet écrit → {output_path}")
    
    out_csv = os.path.splitext(output_path)[0] + ".csv"
    df_summary.to_csv(out_csv, index=False)


def main():
    if len(sys.argv) != 5:
        print("Usage: spark-submit compute_assets_performance.py "
              "<quadrant_file> <assets_file> <output_predicted_parquet> <output_target_parquet>")
        sys.exit(1)
        
    quadrant_file  = sys.argv[1]
    assets_file    = sys.argv[2]
    out_predicted  = sys.argv[3]
    out_target     = sys.argv[4]

    # 1. Load Quadrants
    ext_q = os.path.splitext(quadrant_file)[1].lower()
    if ext_q == '.parquet':
        df_quadrant = pd.read_parquet(quadrant_file)
    else:
        df_quadrant = pd.read_csv(quadrant_file, parse_dates=['date'])
    df_quadrant['date'] = pd.to_datetime(df_quadrant['date'])

    # 2. Load Assets & Pre-calculate Returns (once)
    df_assets_wide = pd.read_parquet(assets_file)
    asset_columns = [c for c in df_assets_wide.columns if c != 'date']
    
    if len(asset_columns) == 0:
        raise ValueError("Aucune colonne d'actif détectée.")
    
    print(f"[compute_assets_performance] Actifs analysés : {asset_columns}")

    # Transform to long format
    df_long = df_assets_wide.melt(
        id_vars=['date'],
        value_vars=asset_columns,
        var_name='asset_id',
        value_name='close'
    ).dropna(subset=['close'])

    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long = df_long.sort_values(['asset_id', 'date'])
    df_long['ret'] = df_long.groupby('asset_id')['close'].pct_change()
    
    # 3. Compute PREDICTED performance (assigned_quadrant)
    compute_performance(df_long, df_quadrant, 'assigned_quadrant', out_predicted)
    
    # 4. Compute TARGET performance (target_quadrant)
    compute_performance(df_long, df_quadrant, 'target_quadrant', out_target)


def get_carry_adjusted_returns(df_long, indicators_file):
    """
    Adds Daily Carry to asset returns for Forex pairs.
    Daily Carry = (Long Rate - Short Rate) / 100 / 252
    """
    if not os.path.exists(indicators_file):
        print(f"⚠️ Indicators file not found: {indicators_file}. Skipping Carry.")
        return df_long

    print(f"[Carry] Loading indicators from {indicators_file}...")
    df_ind = pd.read_csv(indicators_file, parse_dates=['date'])
    df_ind = df_ind.set_index('date').sort_index()
    
    # Forward fill to handle monthly rates (~30 day lag max)
    df_ind = df_ind.ffill()
    
    # Map Forex Pairs to (Long Rate, Short Rate)
    # Quotes are USD/XXX usually, but YF mapping is:
    # USD_EUR (Long USD, Short EUR) -> if quote is EUR=X (1.05), it's EUR/USD. 
    # WAIT: YF_FOREX_MAPPING in dag: 'USD_EUR': 'USDEUR=X' (0.95 EUR per USD). 
    # If pair is USD/EUR (value of 1 USD in EUR), then Long USD, Short EUR.
    # We earn USD Rate, Pay EUR Rate.
    
    CARRY_MAPPING = {
        'USD_EUR': ('TAUX_FED', 'TAUX_ECB'),
        'USD_JPY': ('TAUX_FED', 'TAUX_BOJ'),
        'USD_CAD': ('TAUX_FED', 'TAUX_BOC'),
        'USD_AUD': ('TAUX_FED', 'TAUX_RBA'),
        'USD_BRL': ('TAUX_FED', 'TAUX_BCB'),
    }
    
    # We need to join rates to df_long
    # df_long has 'date', 'asset_id', 'ret'
    
    # Filter only relevant rates
    rate_cols = set()
    for l, s in CARRY_MAPPING.values():
        rate_cols.add(l)
        rate_cols.add(s)
    
    available_rates = [c for c in rate_cols if c in df_ind.columns]
    if not available_rates:
        print("⚠️ No rate columns found in indicators. Skipping Carry.")
        return df_long
        
    df_rates = df_ind[available_rates].copy()
    
    # Merge rates into df_long
    df_long = df_long.merge(df_rates, on='date', how='left')
    
    # Calculate Carry
    df_long['carry'] = 0.0
    
    for asset, (long_rate, short_rate) in CARRY_MAPPING.items():
        if long_rate in df_long.columns and short_rate in df_long.columns:
            mask = df_long['asset_id'] == asset
            
            # Rate is %, so /100. Daily = /252
            # Handle NaNs in rates (e.g. before 2005) -> 0 carry
            l_r = df_long.loc[mask, long_rate].fillna(0.0)
            s_r = df_long.loc[mask, short_rate].fillna(0.0)
            
            daily_carry = (l_r - s_r) / 100.0 / 252.0
            
            # Add to return
            # Total Return = Price Return + Carry
            df_long.loc[mask, 'ret'] += daily_carry
            df_long.loc[mask, 'carry'] = daily_carry
            
            avg_carry = daily_carry.mean() * 252 * 100
            print(f"   [Carry] Applied to {asset}: Avg Annual Carry Spread = {avg_carry:.2f}%")

    # --- GENERATE INVERSE PAIRS (e.g. BRL_USD) ---
    inverse_frames = []
    
    # Identify unique assets processed
    processed_assets = df_long['asset_id'].unique()
    
    for asset in processed_assets:
        # Check if it is a USD pair (e.g. USD_BRL)
        parts = asset.split('_')
        if len(parts) == 2 and parts[0] == 'USD':
            base, quote = parts[0], parts[1]
            inv_asset = f"{quote}_{base}"  # e.g. BRL_USD
            
            print(f"   [Inverse] Generating {inv_asset} from {asset}...")
            
            # Extract original data for this asset
            mask = df_long['asset_id'] == asset
            df_inv = df_long[mask].copy()
            df_inv['asset_id'] = inv_asset
            
            # Invert Price (If 1 USD = 5 BRL, then 1 BRL = 0.2 USD)
            df_inv['close'] = 1.0 / df_inv['close']
            
            # Recalculate Returns based on Inverted Price
            # Note: We cannot just invert the return. We must recompute pct_change on price.
            df_inv = df_inv.sort_values('date')
            df_inv['ret'] = df_inv['close'].pct_change()
            
            # Invert Carry
            # Original: Earn USD, Pay Foreign -> Carry = (USD - Foreign)
            # Inverse: Earn Foreign, Pay USD -> Carry = (Foreign - USD) = -Original
            # Note: We use the already calculated 'carry' column which is (Long - Short)/100/252
            if 'carry' in df_inv.columns:
                df_inv['carry'] = -df_inv['carry']
            else:
                df_inv['carry'] = 0.0
            
            # Total Return = Price Return + Carry
            df_inv['ret'] = df_inv['ret'].fillna(0) + df_inv['carry'].fillna(0)
            
            inverse_frames.append(df_inv)
            
    if inverse_frames:
        df_inverse = pd.concat(inverse_frames, ignore_index=True)
        df_long = pd.concat([df_long, df_inverse], ignore_index=True)
        print(f"   [Inverse] Added {len(inverse_frames)} inverse currency pairs.")
            
    return df_long


def main():
    if len(sys.argv) < 5:
        print("Usage: spark-submit compute_assets_performance.py "
              "<quadrant_file> <assets_file> <output_predicted_parquet> <output_target_parquet> [indicators_file]")
        sys.exit(1)
        
    quadrant_file  = sys.argv[1]
    assets_file    = sys.argv[2]
    out_predicted  = sys.argv[3]
    out_target     = sys.argv[4]
    indicators_file = sys.argv[5] if len(sys.argv) > 5 else None

    # 1. Load Quadrants
    ext_q = os.path.splitext(quadrant_file)[1].lower()
    if ext_q == '.parquet':
        df_quadrant = pd.read_parquet(quadrant_file)
    else:
        df_quadrant = pd.read_csv(quadrant_file, parse_dates=['date'])
    df_quadrant['date'] = pd.to_datetime(df_quadrant['date'])

    # 2. Load Assets & Pre-calculate Returns (once)
    df_assets_wide = pd.read_parquet(assets_file)
    asset_columns = [c for c in df_assets_wide.columns if c != 'date']
    
    if len(asset_columns) == 0:
        raise ValueError("Aucune colonne d'actif détectée.")
    
    print(f"[compute_assets_performance] Actifs analysés : {asset_columns}")

    # Transform to long format
    df_long = df_assets_wide.melt(
        id_vars=['date'],
        value_vars=asset_columns,
        var_name='asset_id',
        value_name='close'
    ).dropna(subset=['close'])

    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long = df_long.sort_values(['asset_id', 'date'])
    df_long['ret'] = df_long.groupby('asset_id')['close'].pct_change()
    
    # 2b. Apply Carry (if indicators provided)
    if indicators_file:
        df_long = get_carry_adjusted_returns(df_long, indicators_file)
    
    # 3. Compute PREDICTED performance (assigned_quadrant)
    compute_performance(df_long, df_quadrant, 'assigned_quadrant', out_predicted)
    
    # 4. Compute TARGET performance (target_quadrant)
    compute_performance(df_long, df_quadrant, 'target_quadrant', out_target)


if __name__ == "__main__":
    main()
