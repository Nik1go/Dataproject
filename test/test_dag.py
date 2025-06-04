import pandas as pd

dfq = pd.read_parquet("/home/leoja/airflow/data/quadrants.parquet")
print(dfq.shape)       # Nombre de lignes et colonnes
print(dfq.columns)     # Liste des colonnes
print(dfq.head(5))     #  Aperçu des 5 premières lignes