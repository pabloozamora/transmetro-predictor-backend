import os
import pandas as pd
from typing import List
import lightgbm as lgb
import numpy as np

MODEL_PATH = "D:\\2025\\UVG\\Tesis\\repos\\backend\\models\\lightgbm\\final\\lgb_final_model.txt"

CATEGORICAL_FEATURES = ["LINEA","DIR","proxima_est_teorica"]
FLOAT_FEATURES = [
    "dist_a_prox_m","dist_estacion_m","vel_mps","Altitud","s_m","dist_m",
    "time_diff","dwell_same_xy_s"
]
INT_FEATURES = ["hour","dow"]
BOOL_FEATURES = ["is_no_progress","progress_event","is_weekend","is_peak"]
    
def load_model():
    if os.path.isfile(MODEL_PATH):
        return [lgb.Booster(model_file=MODEL_PATH)]
    else:
        raise FileNotFoundError("No se encontró modelo: define LGB_MODEL_DIR o LGB_MODEL_PATH")
    return boosters

def df_from_latest_dict(data: dict) -> pd.DataFrame:
    """
    Convierte un único registro (dict) a DataFrame con columnas en el orden
    y tipos esperados por el modelo. NO renombra columnas.
    """
    
    print('Datos recibidos para predicción:', data)
    
    if not isinstance(data, dict):
        raise TypeError("data debe ser un dict con un solo registro")
    
    df = pd.DataFrame([data]).copy()
    
    # Normalizar nombre de columna con espacio (ej. "Altitud (m)")
    df.columns = df.columns.str.replace(" (m)", "", regex=False)
    
    # Asegura columnas y orden
    missing = [c for c in CATEGORICAL_FEATURES + FLOAT_FEATURES + INT_FEATURES + BOOL_FEATURES if c not in df.columns]
    if missing:
        return None
    df = df[CATEGORICAL_FEATURES + FLOAT_FEATURES + INT_FEATURES + BOOL_FEATURES]
    
    # Asegura tipos
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")
    for c in FLOAT_FEATURES:
        df[c] = df[c].astype("float32")
    for c in INT_FEATURES:
        df[c] = df[c].astype("int32")
    for c in BOOL_FEATURES:
        df[c] = df[c].astype("bool")
    
    return df

def predict_boosters(boosters: List[lgb.Booster], data: dict) -> np.ndarray:
    X = df_from_latest_dict(data)
    if X is None:
        return np.array([])
    preds_list = []
    for b in boosters:
        p = b.predict(X, raw_score=False, num_iteration=b.best_iteration)
        preds_list.append(np.asarray(p).reshape(-1))
    # Promedio de folds, si hubiera
    prediction = np.mean(preds_list, axis=0)
    
    # No devolver negativos
    prediction = np.maximum(prediction, 0.0)
    print('Predicción a estación más cercana:', prediction)
    return prediction