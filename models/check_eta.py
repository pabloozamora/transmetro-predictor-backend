from pathlib import Path
import pandas as pd, numpy as np


def preprocess_check_eta_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa la columna de duración para análisis de ETA."""
    df['ETA_proxima_est_s'] = pd.to_numeric(df['ETA_proxima_est_s'], errors='coerce')
    
    # si la duración es extremadamente alta, imprimir advertencia
    high_duration_threshold = 7200  # 2 horas en segundos
    
    outliers = df[df['ETA_proxima_est_s'] > high_duration_threshold]
    if not outliers.empty:
        print(f"Advertencia: Se encontraron {len(outliers)} filas con duración mayor a {high_duration_threshold} segundos.")
        print(outliers[['trip_id', 'block_id', 'ETA_proxima_est_s']])
    
    return df

# Cargar parquets
FEATS_DIR = Path("features_ready")
files = sorted(FEATS_DIR.glob("*_features.parquet"))
assert files, "No hay archivos en features_ready/*.parquet"

# Cargar un parquet a la vez y preprocesar
for file_path in files:
    print(f"Cargando y preprocesando {file_path}...")
    df = pd.read_parquet(file_path)
    df = preprocess_check_eta_duration(df)
