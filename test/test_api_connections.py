import os
import yfinance as yf
import pandas as pd

# Mapping de nom vers identifiant de fichier
SERIES_MAPPING = {
    'S&P 500': {'ticker': '^GSPC', 'series_id': 'SP500_CLOSE'},
    "Or (once en $)": {'ticker': 'GC=F', 'series_id': 'GOLD_OUNCE_USD'}
}

# Dossier de base
base_dir = './data'

# Boucle sur chaque actif à récupérer
for name, meta in SERIES_MAPPING.items():
    print(f"Téléchargement des données pour {name}...")

    # Télécharger les données
    data = yf.download(meta['ticker'], start='2006-01-01', progress=False)

    # Garder la date et la clôture
    df = data[['Close']].reset_index()
    df.columns = ['date', 'value']
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Préparer le chemin
    backup_path = os.path.join(base_dir, 'backup', f"{meta['series_id']}.csv")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)

    # Enregistrement CSV
    df.to_csv(backup_path, index=False)
    print(f"✅ {name} sauvegardé dans {backup_path}")
