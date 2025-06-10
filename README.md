# Macro‐Trading Airflow + Spark Project

Ce projet et une étude de stratégie contre-cyclique basée sur les quadrants économiques, avec :

°° Airflow pour l’orchestration mensuelle °°
  - Récupération des données macro (FRED) et assets (Yahoo Finance)
  - Nettoyage & mise en forme (mensuel & journalier)
  - Calcul des quadrants (Spark)
  - Calcul de la performance par quadrant (Spark)
  - Backtest de la stratégie fixe (Spark)
  - Indexation dans Elasticsearch (BashOperator + Python)

°° Spark (PySpark + Pandas) pour °°
  - Calcul des quadrants économique  (`compute_quadrants.py`)
  - Calcul des performances d’actifs dans chacun des quadrants  (`compute_assets_performance.py`)
  - Backtest de la stratégie d’allocation d'un portefeuille qui suiverais les meilleur actifs sur chaque quadrant(`backtest_strategy.py`)

°° Elasticsearch & Kibana pour visualiser °°
  - Les quadrants et les performances d’actifs
  - La valeur du portefeuille, SP500 et Gold au fil du temps  
  - Les ratios de Sharpe et autres métriques agrégées
