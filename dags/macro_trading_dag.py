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
    'BARREL PETROL': 'DCOILBRENTEU'}

YF_SERIES_MAPPING = {
    'S&P500(LARGE CAP)': {'ticker': '^GSPC', 'series_id': 'SP500'},
    "GOLD_OZ_USD": {'ticker': 'GC=F', 'series_id': 'GOLD_OZ_USD'},
    "RUSSELL2000(Small CAP)" : {'ticker': 'IWM', 'series_id': 'SmallCAP'},
    "REITs(Immobilier US)": {'ticker': 'VNQ', 'series_id': 'US REIT VNQ'},
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
            start_date = datetime(2006, 1, 1)

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
            start_date = datetime(2006, 1, 1)

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

fetch_and_save_data()

def prepare_combined_data(base_dir):
    """Crée un fichier combiné HY_SPREAD + LONG_SPREAD pour Spark"""
    backup_dir = os.path.join(base_dir, 'backup')

    # Charger HY Spread
    hy_path = os.path.join(backup_dir, 'High_Yield_Bond_SPREAD.csv')
    hy_df = pd.read_csv(hy_path, parse_dates=['date']) if os.path.exists(hy_path) else pd.DataFrame(
        columns=['date', 'value'])
    hy_df = hy_df.rename(columns={'value': 'High_Yield_Bond_SPREAD'})

    # Charger Long Spread
    long_path = os.path.join(backup_dir, '10-2Year_Treasury_Yield_Bond')
    long_df = pd.read_csv(long_path, parse_dates=['date']) if os.path.exists(long_path) else pd.DataFrame(
        columns=['date', 'value'])
    long_df = long_df.rename(columns={'value': '10-2Year_Treasury_Yield_Bond'})

    # Fusionner et sauvegarder
    combined = pd.merge(hy_df, long_df, on='date', how='outer').sort_values('date')
    combined_path = os.path.join(base_dir, 'combined_spreads.csv')
    combined.to_csv(combined_path, index=False)
    print(f"Fichier combiné créé: {combined_path}")


def run_strategy_with_spark(**kwargs):
    """Exécute la stratégie avec Spark"""
    base_dir = os.path.expanduser('~/airflow/data')
    input_path = os.path.join(base_dir, 'combined_spreads.csv')

    spark_args = {
        'input_path': input_path,
        'output_path': os.path.join(base_dir, 'results')
    }

    return SparkSubmitOperator(
        task_id='spark_strategy_task',
        application="home/leoja/airflow/scripts/spark_strategy.py",
        application_args=[
            '--input', spark_args['input_path'],
            '--output', spark_args['output_path']
        ],
        conn_id="spark_default",
        dag=kwargs['dag']
    ).execute(context=kwargs)

# === Configuration du DAG ===

base_dir = os.path.expanduser('~/airflow/data')

with DAG(
    dag_id='macro_trading_dag',
    default_args=default_args,
    description='Stratégie contre-cyclique avec données FRED',
    schedule_interval='0 8 * * *',
    catchup=False
) as dag:

    fetch_task = PythonOperator(
        task_id='fetch_fred_data',
        python_callable=fetch_and_save_data
    )

    combine_task = PythonOperator(
        task_id='prepare_combined_data',
        python_callable=prepare_combined_data,
        op_args=[base_dir]
    )

    strategy_task = SparkSubmitOperator(
        task_id='run_countercyclical_strategy',
        application='/home/leoja/airflow/scripts/spark_strategy.py',
        application_args=[
            '--input', os.path.join(base_dir, 'combined_spreads.csv'),
            '--output', os.path.join(base_dir, 'results')
        ],
        conn_id='spark_default'
    )

    # Enchaînement des tâches
    fetch_task >> combine_task >> strategy_task