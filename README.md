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
 
 ⚙️ Prérequis

- **Python 3.8+** (venv recommandé)  
- **Apache Airflow** (2.x) avec Spark provider  
- **Apache Spark** (3.x) et `spark-submit`  
- **Elasticsearch** (8.x) + **Kibana**  
- Bibliothèques Python : `pandas`, `numpy`, `fredapi`, `yfinance`, `requests`, `elasticsearch`

- ## 🚀 Installation & déploiement

1. **Cloner le dépôt** dans `$AIRFLOW_HOME` (ex. `~/airflow`).  
2. **Créer un virtualenv** et installer les dépendances :
   ```bash
   cd ~/airflow
   python3 -m venv airflow_venv
   source airflow_venv/bin/activate
   pip install apache-airflow apache-airflow-providers-apache-spark pyspark pandas numpy fredapi yfinance requests elasticsearch
   Airflow standalone


   PS: Ce script est pour les PNL MAKER
   
