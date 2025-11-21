from pathlib import Path
import pandas as pd, numpy as np


def preprocess_check_eta_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa la columna de duración para análisis de ETA."""
    df['ETA_proxima_est_s'] = pd.to_numeric(df['ETA_proxima_est_s'], errors='coerce')
    
    # si la duración es extremadamente alta, imprimir advertencia
    high_duration_threshold = 7200  # 2 horas en segundos
    
    outliers = df[df['ETA_proxima_est_s'] > high_duration_threshold]
    if not outliers.empty:
        print(f'Advertencia: Se encontraron duraciones de ETA extremadamente altas (> 7200 segundos) en el trip_id(s): {outliers["trip_id"].unique()}')
    
    return df

# Cargar parquets
FEATS_DIR = Path("D:/2025/UVG/Tesis/repos/backend/compact_datasets/")
files = sorted(FEATS_DIR.glob("*.parquet"))
assert files, "No hay archivos en compact_datasets/*.parquet"

# Cargar un parquet a la vez y preprocesar
for file_path in files:
    print(f"Procesando unidad: {file_path.name.split('_')[0]}")
    df = pd.read_parquet(file_path)
    df = preprocess_check_eta_duration(df)
