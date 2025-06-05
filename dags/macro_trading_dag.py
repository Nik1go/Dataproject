from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import yfinance as yf
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from fredapi import Fred

FRED_API_KEY = 'c4caaa1267e572ae636ff75a2a600f3d'

FRED_SERIES_MAPPING = {
    'INFLATION': 'CPIAUCSL',
    'UNEMPLOYMENT': 'UNRATE',
    'High_Yield_Bond_SPREAD': 'BAMLH0A0HYM2',
    '10-2Year_Treasury_Yield_Bond': 'T10Y2Y',
    'CONSUMER_SENTIMENT': 'UMCSENT',
    'BARREL PETROL': 'DCOILBRENTEU',
    'TAUX_FED': 'FEDFUNDS'}


YF_SERIES_MAPPING = {
    'S&P500(LARGE CAP)': {'ticker': '^GSPC', 'series_id': 'SP500'},
    "GOLD_OZ_USD": {'ticker': 'GC=F', 'series_id': 'GOLD_OZ_USD'},
    "RUSSELL2000(Small CAP)": {'ticker': 'IWM', 'series_id': 'SmallCAP'},
    "REITs(Immobilier US)": {'ticker': 'VNQ', 'series_id': 'US_REIT_VNQ'},
    'US_TREASURY_10Y': {'ticker': 'IEF', 'series_id': 'TREASURY_10Y'},
}

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3)
}


def fetch_and_save_data(**kwargs):
    fred = Fred(api_key=FRED_API_KEY)
    base_dir = os.path.expanduser('~/airflow/data')

    # --- Données FRED ---
    for name, series_id in FRED_SERIES_MAPPING.items():
        backup_path = os.path.join(base_dir, 'backup', f'{name}.csv')
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        existing_data = pd.DataFrame()
        if os.path.exists(backup_path):
            existing_data = pd.read_csv(backup_path, parse_dates=['date'])
            last_date = existing_data['date'].max()
            start_date = last_date + pd.Timedelta(days=1)
        else:
            start_date = datetime(2005, 1, 1)

        new_data = fred.get_series(series_id, observation_start=start_date)

        if not new_data.empty:
            new_df = new_data.reset_index()
            new_df.columns = ['date', 'value']
            new_df['date'] = pd.to_datetime(new_df['date']).dt.date

            if not existing_data.empty:
                existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
                combined = pd.concat([existing_data, new_df])
                combined = combined.drop_duplicates('date').sort_values('date')
            else:
                combined = new_df

            combined.to_csv(backup_path, index=False)
            print(f'Données mises à jour pour {name} ({series_id})')
        else:
            print(f'Aucune nouvelle donnée pour {name} ({series_id})')

    # --- Données Yahoo Finance ---
    for name, meta in YF_SERIES_MAPPING.items():
        backup_path = os.path.join(base_dir, 'backup', f"{name}.csv")
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

        existing_data = pd.DataFrame()
        if os.path.exists(backup_path):
            existing_data = pd.read_csv(backup_path, parse_dates=['date'])
            last_date = existing_data['date'].max()
            start_date = last_date + pd.Timedelta(days=1)
        else:
            start_date = datetime(2005, 1, 1)

        end_date = datetime.today() - timedelta(days=1)
        start_date = min(start_date, end_date)

        if start_date.date() > end_date.date():
            print(f"Pas de nouvelles données à récupérer pour {name} ({meta['series_id']})")
            continue

        data = yf.download(meta['ticker'], start=start_date, end=end_date, progress=False, auto_adjust=True)

        if not data.empty:
            df = data[['Close']].reset_index()
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date']).dt.date

            if not existing_data.empty:
                existing_data['date'] = pd.to_datetime(existing_data['date']).dt.date
                combined = pd.concat([existing_data, df])
                combined = combined.drop_duplicates('date').sort_values('date')
            else:
                combined = df

            combined.to_csv(backup_path, index=False)
            print(f"Données mises à jour pour {name} ({meta['series_id']})")
        else:
            print(f"Aucune nouvelle donnée pour {name} ({meta['series_id']})")


def prepare_indicators_data(base_dir):
    """Combine les indicateurs économiques en un seul DataFrame"""
    backup_dir = os.path.join(base_dir, 'backup')
    indicators = [
        'INFLATION',
        'UNEMPLOYMENT',
        'CONSUMER_SENTIMENT',
        'High_Yield_Bond_SPREAD',
        '10-2Year_Treasury_Yield_Bond',
        'TAUX_FED'

    ]

    combined_df = pd.DataFrame()

    for indicator in indicators:
        file_path = os.path.join(backup_dir, f"{indicator}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['date'])
            df = df.rename(columns={'value': indicator})

            if combined_df.empty:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='date', how='outer')

    # Sauvegarder temporairement
    output_path = os.path.join(base_dir, 'combined_indicators.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Fichier combiné des indicateurs créé: {output_path}")
    return output_path


def prepare_assets_data(base_dir):
    """Combine les actifs en un seul DataFrame"""
    backup_dir = os.path.join(base_dir, 'backup')
    assets = list(YF_SERIES_MAPPING.keys())

    combined_df = pd.DataFrame()

    for asset in assets:
        file_path = os.path.join(backup_dir, f"{asset}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['date'])
            # Récupérer le nom court de l'actif
            asset_name = YF_SERIES_MAPPING[asset]['series_id']
            df = df.rename(columns={'value': asset_name})

            if combined_df.empty:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='date', how='outer')

    # Sauvegarder temporairement
    output_path = os.path.join(base_dir, 'combined_assets.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Fichier combiné des actifs créé: {output_path}")
    return output_path


def format_and_clean_data(base_dir, input_path, data_type):
    print(f"→ format_and_clean_data: on lit le fichier CSV : {input_path}")

    # Lire les données
    df = pd.read_csv(input_path, parse_dates=['date'])

    print("   Colonnes lues dans df :", df.columns.tolist())

    # Supprimer les lignes où toutes les valeurs sont nulles
    df = df.dropna(how='all', subset=df.columns.difference(['date']))

    # Convertir en mensuel (dernier jour du mois)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    monthly_df = df.resample('M').last()

    # Interpolation pour les valeurs manquantes
    monthly_df = monthly_df.interpolate(method='linear')

    # Réinitialiser l'index
    monthly_df.reset_index(inplace=True)

    # → **Ici** : convertir la colonne `date` en string (YYYY-MM-DD)
    monthly_df['date'] = monthly_df['date'].dt.strftime('%Y-%m-%d')

    # Définir le chemin de sortie
    output_path = os.path.join(base_dir, f"{data_type}.parquet")

    # Sauvegarder en Parquet
    monthly_df.to_parquet(output_path, index=False)
    print(f"Données {data_type} mensuelles nettoyées sauvegardées: {output_path}")

    # Aperçu
    print(monthly_df.tail(5))

    return output_path


def format_and_clean_data_daily(base_dir, input_path, data_type):
    """
    - Lit le CSV journalier dont le chemin est input_path
    - Supprime les jours où il manque au moins une donnée (colonnes autres que 'date')
    - Trie par date et convertit la date en format YYYY-MM-DD (string)
    - Écrit le DataFrame nettoyé en Parquet
    """
    print(f"→ format_and_clean_data_daily: on lit le fichier CSV : {input_path}")

    # 1. Lecture du CSV et conversion de la colonne 'date' en datetime
    df = pd.read_csv(input_path, parse_dates=['date'])
    print("   Colonnes lues dans df :", df.columns.tolist())

    # 2. Supprimer les lignes où TOUTES les colonnes (sauf 'date') sont nulles
    df = df.dropna(how='all', subset=df.columns.difference(['date']))

    # 3. Supprimer les lignes où AU MOINS UNE colonne (autre que 'date') est nulle
    #    → Ainsi on retire purement et simplement tous les jours qui contiennent un ou plusieurs NULLs
    df = df.dropna(how='any', subset=df.columns.difference(['date']))

    # 4. S’assurer que la colonne 'date' est bien un datetime, puis trier
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 5. (Optionnel) Si vous aviez besoin de réindexes sur les week-ends, etc.,
    #    vous pourriez le faire ici, mais dans notre cas on conserve strictement
    #    uniquement les dates présentes dans le CSV.

    # 6. Convertir la date en string "YYYY-MM-DD" (c’est souvent plus simple pour lire le Parquet)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # 7. Sauvegarde en Parquet
    output_path = os.path.join(base_dir, f"{data_type}_daily.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Données {data_type} journalières nettoyées sauvegardées : {output_path}")

    # 8. Affichage d’un petit aperçu
    print(df.head(5))
    print(df.tail(5))

    return output_path

# === Configuration du DAG ===

base_dir = os.path.expanduser('~/airflow/data')

with (DAG(
        dag_id='macro_trading_dag',
        default_args=default_args,
        description='Stratégie contre-cyclique avec données FRED et Yahoo Finance',
        schedule_interval='0 8 * * *',
        catchup=False,
        tags=['macro','assets','performance']
) as dag):
    fetch_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_and_save_data
    )

    prepare_indicators_task = PythonOperator(
        task_id='prepare_indicators_data',
        python_callable=prepare_indicators_data,
        op_kwargs={'base_dir': base_dir}
    )

    prepare_assets_task = PythonOperator(
        task_id='prepare_assets_data',
        python_callable=prepare_assets_data,
        op_kwargs={'base_dir': base_dir}
    )

    format_indicators_task = PythonOperator(
        task_id='format_indicators_data',
        python_callable=format_and_clean_data,
        op_kwargs={
            'base_dir': base_dir,
            'input_path': "{{ ti.xcom_pull(task_ids='prepare_indicators_data') }}",
            'data_type': 'Indicators'
        }
    )

    format_assets_task = PythonOperator(
        task_id='format_assets_data',
        python_callable=format_and_clean_data_daily,
        op_kwargs={
            'base_dir': base_dir,
            'input_path': "{{ ti.xcom_pull(task_ids='prepare_assets_data') }}",
            'data_type': 'Assets'
        }
    )
    ASSETS_PERF_OUTPUT = os.path.join(base_dir, "assets_performance_by_quadrant.parquet")

    INDICATORS_PARQUET = os.path.join(base_dir, "Indicators.parquet")
    QUADRANT_OUTPUT = os.path.join(base_dir, "quadrants.parquet")
    QUADRANT_CSV = os.path.join(base_dir, "data/quadrants.csv")

    compute_quadrant_task = SparkSubmitOperator(
        task_id='compute_economic_quadrants',
        application="/home/leoja/airflow/spark_jobs/compute_quadrants.py",
        name="compute_economic_quadrants",
        application_args=[INDICATORS_PARQUET, QUADRANT_OUTPUT, QUADRANT_CSV],
        conn_id="spark_local",  # on reste sur yank ‘spark_local’
        conf={
            "spark.pyspark.python": "/home/leoja/airflow_venv/bin/python",
            "spark.pyspark.driver.python": "/home/leoja/airflow_venv/bin/python"
        },
        verbose=False
    )
    compute_assets_performance_task = SparkSubmitOperator(
        task_id='compute_assets_performance',
        application="/home/leoja/airflow/spark_jobs/compute_assets_performance.py",
        name="compute_assets_performance",
        application_args=[
            # 1) Chemin vers quadrant.parquet   (écrit par compute_economic_quadrants)
            QUADRANT_OUTPUT,

            # 2) Chemin vers Assets_daily.parquet (écrit par format_assets_data)
            "{{ ti.xcom_pull(task_ids='format_assets_data') }}",

            # 3) Chemin de sortie final assets_performance_by_quadrant.parquet
            ASSETS_PERF_OUTPUT
        ],
        conn_id="spark_local",
        conf={
            "spark.pyspark.python": "/home/leoja/airflow_venv/bin/python",
            "spark.pyspark.driver.python": "/home/leoja/airflow_venv/bin/python"
        },
        verbose=False
    )
    # → Puisque compute_assets_performance.py est un script Spark,
    #   on l’appelle avec spark-submit, exactement comme pour compute_quadrants.py.

    # ── Étape 6: CHAÎNAGE DES TÂCHES ──────────────────────────────────────────────

    # 1 → 2 & 3 : on fetch les données, puis on prépare assets + indicators
    fetch_task >> [prepare_indicators_task, prepare_assets_task]

    # 2 → 4 : format des indicateurs → job Spark quadrants
    prepare_indicators_task >> format_indicators_task >> compute_quadrant_task

    # 3 : format des assets quotidiens après prepare_assets_task
    prepare_assets_task >> format_assets_task

    # 4 + 3 → 5 : on attend à la fois
    #    • le job Spark des quadrants (compute_quadrant_task)
    #    • et le format quotidien des assets (format_assets_task)
    [compute_quadrant_task, format_assets_task] >> compute_assets_performance_task
