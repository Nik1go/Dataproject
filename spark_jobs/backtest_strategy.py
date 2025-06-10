
import os
import sys
import shutil
import pandas as pd
import numpy as np

"""
backtest_strategy.py

Ce script réalise un backtest mensuel d’une stratégie de rotation d’actifs basée sur les quadrants
économiques. Pour chaque mois :

1. Lecture du quadrant courant depuis “quadrants.csv”.
2. Calcul des rendements mensuels de fin de mois pour les actifs SP500, Gold, SmallCap, REITs et Treasuries
   à partir du fichier “Assets_daily.parquet”.
3. Application d’une allocation fixe par quadrant :
   - Q1 : 50 % SP500, 40 % SmallCap, 10 % REITs
   - Q2 : 50 % Treasuries, 40 % Gold,    10 % REITs
   - Q4 : 50 % Gold,     40 % Treasuries,10 % SmallCap
   - Q3 : 50 % Gold,     40 % SP500,     10 % REITs
4. Simulation de la valeur du portefeuille à partir d’un capital initial (ex. 1000 $), cumul des rendements.
5. Calcul des métriques de performance pour la stratégie, le SP500 et le Gold :
   - Volatilité annualisée
   - Ratio de Sharpe annualisé
   - Drawdown maximal
   - Rendement moyen annualisé
6. Génération des fichiers de sortie dans “backtest_results/” :
   - backtest_timeseries.csv / .parquet : série mensuelle de rendements et de valeur du portefeuille
   - backtest_stats.csv             : statistiques agrégées de performance

Usage (via spark-submit ou python) :
    backtest_strategy.py <quadrants.csv> <Assets_daily.parquet> <capital_initial> <dossier_sortie>
"""

def compute_monthly_returns_from_parquet(parquet_path, assets):

    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.set_index('date').resample('ME').last().ffill()
    rets = monthly[assets].pct_change().rename(columns=lambda c: f"{c}_ret")
    rets.index = rets.index.to_period('M').to_timestamp()
    return rets

def max_drawdown(wealth_series):
    rm = wealth_series.cummax()
    return ((rm - wealth_series) / rm).max()

def stats_from_series(ret_series, wealth_series):

    mean_r   = ret_series.mean()
    std_r    = ret_series.std(ddof=1)
    sharpe_m = mean_r / std_r if std_r > 0 else np.nan
    sharpe_a = sharpe_m * np.sqrt(12)
    vol_a    = std_r * np.sqrt(12)
    md       = max_drawdown(wealth_series)
    avg_yr   = mean_r * 12
    return vol_a, sharpe_a, md, avg_yr

def main():
    if len(sys.argv) != 5:
        print("Usage: backtest_strategy.py "
              "<quadrants.csv> <Assets_daily.parquet> <initial_capital> <output_dir>")
        sys.exit(1)

    quadrants_csv, assets_parquet, initial_capital, output_dir = sys.argv[1:]
    initial_capital = float(initial_capital)

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_q = pd.read_csv(quadrants_csv, parse_dates=['date'])
    df_q['year_month'] = df_q['date'].dt.to_period('M').dt.to_timestamp()
    df_q = df_q.set_index('year_month')['assigned_quadrant']

    assets = ['SP500','GOLD_OZ_USD','SmallCAP','US_REIT_VNQ','TREASURY_10Y']
    df_rets = compute_monthly_returns_from_parquet(assets_parquet, assets)

    weights = {
        1: {'SP500':0.5, 'SmallCAP':0.4,    'US_REIT_VNQ':0.1},
        2: {'TREASURY_10Y':0.5, 'GOLD_OZ_USD':0.4, 'US_REIT_VNQ':0.1},
        4: {'GOLD_OZ_USD':0.5,  'TREASURY_10Y':0.4, 'SmallCAP':0.1},
        3: {'GOLD_OZ_USD':0.5,  'SP500':0.4,       'US_REIT_VNQ':0.1}
    }

    records = []
    for ym, quad in df_q.items():
        row = {'year_month': ym}

        row['SP500_ret']       = df_rets.at[ym, 'SP500_ret']      if ym in df_rets.index else np.nan
        row['GOLD_OZ_USD_ret'] = df_rets.at[ym, 'GOLD_OZ_USD_ret'] if ym in df_rets.index else np.nan

        if ym in df_rets.index:
            rents = df_rets.loc[ym]
            alloc = weights.get(int(quad), {})
            port_ret = sum(
                rents[f"{asset}_ret"] * w
                for asset, w in alloc.items()
                if f"{asset}_ret" in rents
            )
        else:
            port_ret = 0.0

        row['portfolio_return'] = port_ret
        records.append(row)

    df_bt = pd.DataFrame(records)
    df_bt['date'] = df_bt['year_month'].dt.to_period('M').dt.end_time.dt.strftime('%Y-%m-%d')

    df_bt['date'] = pd.to_datetime(df_bt['date'], format='%Y-%m-%d')
    df_bt = df_bt.set_index('date').sort_index()
    # 6) Simuler la valeur du portefeuille
    df_bt['wealth']        = initial_capital * (1 + df_bt['portfolio_return']).cumprod()
    df_bt['SP500_wealth']  = initial_capital * (1 + df_bt['SP500_ret']).cumprod()
    df_bt['GOLD_wealth']   = initial_capital * (1 + df_bt['GOLD_OZ_USD_ret']).cumprod()

    stats = {}
    for label, (rcol, wcol) in {
        'strategy':('portfolio_return','wealth'),
        'SP500'   :('SP500_ret','SP500_wealth'),
        'GOLD'    :('GOLD_OZ_USD_ret','GOLD_wealth')
    }.items():
        vol_a, sharpe_a, md, avg_yr = stats_from_series(df_bt[rcol], df_bt[wcol])
        stats[f"{label}_vol_annual"]      = vol_a
        stats[f"{label}_sharpe_annual"]   = sharpe_a
        stats[f"{label}_max_drawdown"]    = md
        stats[f"{label}_avg_year_return"] = avg_yr

    df_bt.to_parquet(f"{output_dir}/backtest_timeseries.parquet", index=True)
    df_bt.to_csv(    f"{output_dir}/backtest_timeseries.csv", date_format='%Y-%m-%d')
    pd.DataFrame([stats]).to_csv(f"{output_dir}/backtest_stats.csv", index=False)

    print("Backtest terminé. Stats :", stats)


if __name__ == "__main__":
    main()
