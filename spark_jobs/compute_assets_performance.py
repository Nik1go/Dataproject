
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
spark-submit compute_assets_performance.py <quadrants.parquet> <assets_daily.parquet> <output_summary.parquet> """

def main():
    if len(sys.argv) != 4:
        print("Usage: spark-submit compute_assets_performance.py "
              "<quadrant_file> <assets_file> <output_parquet>")
        sys.exit(1)
    quadrant_file  = sys.argv[1]
    assets_file    = sys.argv[2]
    output_parquet = sys.argv[3]
    parent_out = os.path.dirname(output_parquet)
    if parent_out and not os.path.isdir(parent_out):
        os.makedirs(parent_out, exist_ok=True)

    ext_q = os.path.splitext(quadrant_file)[1].lower()
    if ext_q == '.parquet':
        df_quadrant = pd.read_parquet(quadrant_file)
    elif ext_q in ('.csv', '.txt'):
        df_quadrant = pd.read_csv(quadrant_file, parse_dates=['date'])
    else:
        raise ValueError(f"Extension non supportée pour quadrant_file → {ext_q!r}")

    if 'assigned_quadrant' not in df_quadrant.columns:
        raise KeyError("La colonne 'assigned_quadrant' est absente.")

    df_quadrant['date'] = pd.to_datetime(df_quadrant['date'])
    df_quadrant['year_month'] = df_quadrant['date'].dt.to_period('M').dt.to_timestamp()
    df_q = df_quadrant[['year_month', 'assigned_quadrant']].drop_duplicates()

    df_assets_wide = pd.read_parquet(assets_file)
    if 'date' not in df_assets_wide.columns:
        raise KeyError("La colonne 'date' est absente de Assets_daily.parquet.")
    asset_columns = [c for c in df_assets_wide.columns if c != 'date']
    if len(asset_columns) == 0:
        raise ValueError("Aucune colonne d’actif détectée (hormis 'date').")

    df_long = df_assets_wide.melt(
        id_vars=['date'],
        value_vars=asset_columns,
        var_name='asset_id',
        value_name='close'
    ).dropna(subset=['close'])

    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['year_month'] = df_long['date'].dt.to_period('M').dt.to_timestamp()

    df_long = df_long.sort_values(['asset_id', 'date'])
    df_long['ret'] = df_long.groupby('asset_id')['close'].pct_change()

    df_merged = pd.merge(
        left  = df_long,
        right = df_q,
        on    = 'year_month',
        how   = 'inner'
    )

    rows = []
    grouped = df_merged.groupby(['asset_id', 'assigned_quadrant'])
    for (asset, quadrant), sub in grouped:
        sub = sub.sort_values('date')
        daily_ret = sub['ret'].dropna()
        if len(daily_ret) < 1:
            continue

        first_close    = sub['close'].iloc[0]
        last_close     = sub['close'].iloc[-1]
        monthly_return = (last_close / first_close) - 1

        cumprod     = (1 + daily_ret).cumprod()
        rolling_max = cumprod.cummax()
        drawdown    = (cumprod - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        mean_ret  = daily_ret.mean()
        std_ret   = daily_ret.std()
        sharpe_annualized = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else np.nan
        sharpe_mensuel = sharpe_monthly = mean_ret / std_ret if std_ret > 0 else np.nan

        rows.append({
            'asset_id'           : asset,
            'assigned_quadrant'  : quadrant,
            'start_date'         : sub['date'].min(),
            'end_date'           : sub['date'].max(),
            'monthly_return'     : monthly_return,
            'max_drawdown'       : max_dd,
            'sharpe_annualized'  : sharpe_annualized,
            'sharpe mensuel'     : sharpe_monthly

        })

    df_summary = pd.DataFrame(rows)

    df_summary.to_parquet(output_parquet, index=False)
    print(f"[compute_assets_performance] Parquet écrit → {output_parquet}")
    out_csv = os.path.splitext(output_parquet)[0] + ".csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"[compute_assets_performance] CSV écrit     → {out_csv}")


if __name__ == "__main__":
    main()
