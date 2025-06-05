# ── spark_jobs/compute_assets_performance.py ─────────────────────────────────

import os
import sys
import pandas as pd
import numpy as np

def main():
    # 1) Arguments : <quadrant_file> <assets_file> <output_parquet>
    if len(sys.argv) != 4:
        print("Usage: spark-submit compute_assets_performance.py "
              "<quadrant_file> <assets_file> <output_parquet>")
        sys.exit(1)

    quadrant_file  = sys.argv[1]
    assets_file    = sys.argv[2]
    output_parquet = sys.argv[3]

    # 2) Créer le dossier parent si besoin
    parent_out = os.path.dirname(output_parquet)
    if parent_out and not os.path.isdir(parent_out):
        os.makedirs(parent_out, exist_ok=True)

    # 3) Lire le fichier “quadrant” (Parquet ou CSV)
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

    # 4) Lire le Parquet “wide” des assets
    df_assets_wide = pd.read_parquet(assets_file)
    if 'date' not in df_assets_wide.columns:
        raise KeyError("La colonne 'date' est absente de Assets_daily.parquet.")

    # 5) Déterminer automatiquement les colonnes d’actifs (toutes sauf 'date')
    asset_columns = [c for c in df_assets_wide.columns if c != 'date']
    if len(asset_columns) == 0:
        raise ValueError("Aucune colonne d’actif détectée (hormis 'date').")

    # 6) Melt : wide → long (asset_id, close)
    df_long = df_assets_wide.melt(
        id_vars=['date'],
        value_vars=asset_columns,
        var_name='asset_id',
        value_name='close'
    ).dropna(subset=['close'])

    # 7) Convertir date en datetime → year_month
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['year_month'] = df_long['date'].dt.to_period('M').dt.to_timestamp()

    # 8) Rendements journaliers par actif
    df_long = df_long.sort_values(['asset_id', 'date'])
    df_long['ret'] = df_long.groupby('asset_id')['close'].pct_change()

    # 9) Fusion avec df_q sur 'year_month'
    df_merged = pd.merge(
        left  = df_long,
        right = df_q,
        on    = 'year_month',
        how   = 'inner'
    )

    # 10) Agrégation par (asset_id, assigned_quadrant)
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

        rows.append({
            'asset_id'           : asset,
            'assigned_quadrant'  : quadrant,
            'start_date'         : sub['date'].min(),
            'end_date'           : sub['date'].max(),
            'monthly_return'     : monthly_return,
            'max_drawdown'       : max_dd,
            'sharpe_annualized'  : sharpe_annualized
        })

    df_summary = pd.DataFrame(rows)

    # 11) Écriture overwrite en Parquet + CSV
    df_summary.to_parquet(output_parquet, index=False)
    print(f"[compute_assets_performance] Parquet écrit → {output_parquet}")
    out_csv = os.path.splitext(output_parquet)[0] + ".csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"[compute_assets_performance] CSV écrit     → {out_csv}")


if __name__ == "__main__":
    main()
