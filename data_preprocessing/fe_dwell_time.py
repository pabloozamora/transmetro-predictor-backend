import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ======= Utilidades geodésicas ========

def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Aproximación precisa de metros por grado en la latitud dada."""
    lat = np.deg2rad(lat_deg)
    mlat = 111132.92 - 559.82*np.cos(2*lat) + 1.175*np.cos(4*lat) - 0.0023*np.cos(6*lat)
    mlon = 111412.84*np.cos(lat) - 93.5*np.cos(3*lat) + 0.118*np.cos(5*lat)
    return float(mlat), float(mlon)

def ll_to_xy_m(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convierte lat/lon a x,y en metros usando proyección equirectangular alrededor de (lat0,lon0)."""
    mlat, mlon = meters_per_degree(lat0)
    x = (lon - lon0) * mlon
    y = (lat - lat0) * mlat
    return x.astype(float), y.astype(float)

# =========================
# Dwell por no-avance en lat/lon (convertido a metros)
# =========================
def compute_dwell_same_ll_for_group(g: pd.DataFrame,
                                   min_progress_m: float,
                                   roll_med_win: int,
                                   use_abs_progress: bool) -> pd.DataFrame:
    """
    Calcula dwell por no-progreso en lat y lon:
      - dwell_same_s_s: tiempo (s) acumulado sin progreso suficiente en s.
      - progress_event: 1 cuando hay progreso (resetea el dwell).
      - is_no_progress: 1 si está en estado sin progreso.
    """
    g = g.copy()

    # Asegura tipos y orden temporal
    g['Fecha'] = pd.to_datetime(g['Fecha'], errors="coerce")
    g['Latitud'] = pd.to_numeric(g['Latitud'], errors="coerce")
    g['Longitud'] = pd.to_numeric(g['Longitud'], errors="coerce")
    g = g.sort_values('Fecha', kind="mergesort").reset_index(drop=True)

    # Proyección a metros (referencia = lat/lon del primer punto o la mediana del grupo)
    lat0 = float(g['Latitud'].iloc[0]) if pd.notna(g['Latitud'].iloc[0]) else float(g['Latitud'].median())
    lon0 = float(g['Longitud'].iloc[0]) if pd.notna(g['Longitud'].iloc[0]) else float(g['Longitud'].median())
    x, y = ll_to_xy_m(g['Latitud'].values, g['Longitud'].values, lat0, lon0)

    # Estado e integración
    n = len(g)
    dwell = np.zeros(n, dtype=float)
    is_no_prog = np.zeros(n, dtype=int)
    prog_evt = np.zeros(n, dtype=int)

    for i in range(1, n):
        dt = (g['Fecha'].iloc[i] - g['Fecha'].iloc[i-1]).total_seconds()
        if dt <= 0 or pd.isna(dt):
            dt = 0.0

        ds = np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
        if pd.isna(ds):
            ds = 0.0

        progress = abs(ds) if use_abs_progress else ds

        if progress >= min_progress_m:
            # Hubo progreso suficiente
            dwell[i] = 0.0
            is_no_prog[i] = 0
            prog_evt[i] = 1
        else:
            # No hubo progreso suficiente
            dwell[i] = dwell[i-1] + dt
            is_no_prog[i] = 1
            prog_evt[i] = 0

    g["dwell_same_xy_s"] = dwell
    g["is_no_progress"]  = is_no_prog
    g["progress_event"]  = prog_evt
    return g

# =========================
# Script principal
# =========================
def process_unit(unit):
    
    UNIT = unit
    
    INPUT = f'D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{UNIT}\\{UNIT}_trips_with_next_station.csv'
    OUTPUT = INPUT

    df = pd.read_csv(INPUT)

    # Validaciones mínimas
    if 'Fecha' not in df.columns:
        raise ValueError(f"No se encontró la columna de tiempo 'Fecha'.")
    if 's_m' not in df.columns:
        raise ValueError(f"No se encontró la columna de distancia a lo largo de ruta 's_m'.")

    # Determina agrupación
    group_cols = ['trip_id', 'block_id']
    sort_cols = group_cols + ['Fecha'] if group_cols else ['Fecha']
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Aplica por grupo
    df_out = df.groupby(group_cols, group_keys=False).apply(
        compute_dwell_same_ll_for_group,
        min_progress_m=MIN_PROGRESS_M,
        roll_med_win=ROLL_MED_WIN,
        use_abs_progress=USE_ABS_PROGRESS,
    )

    # Guarda
    df_out.to_csv(OUTPUT, index=False)
    print(f"Guardado en: {OUTPUT}")
    print(f"group_cols: {group_cols if group_cols else '(none)'} | MIN_PROGRESS_M={MIN_PROGRESS_M} m | ROLL_MED_WIN={ROLL_MED_WIN}")

# --- Ejecución principal ---

# Lógica de no-progreso
MIN_PROGRESS_M      = 8.0     # se considera "hubo progreso" si |Δs| >= este umbral
ROLL_MED_WIN        = 3       # mediana móvil (en muestras) para suavizar s_m
USE_ABS_PROGRESS    = True    # True: usa |Δs|

# --- Ejecución principal ---
if __name__ == "__main__":
    print('=== Iniciando cálculo de tiempo de permanencia (dwell time) ===')
    
    # Encontrar todas las unidades (carpetas en data with features)
    DATA_WITH_FEATURES_DIR = Path("D:/2025/UVG/Tesis/repos/backend/data_with_features")
    if not DATA_WITH_FEATURES_DIR.exists():
        print(f"No existe el directorio {DATA_WITH_FEATURES_DIR}. Fin.")
    units = [p.name for p in DATA_WITH_FEATURES_DIR.iterdir() if p.is_dir() and p.name != "maps"]
    
    if not units:
        print(f"No se encontraron unidades en {DATA_WITH_FEATURES_DIR}. Fin.")
        
    for unit in units:
        print(f"--- Procesando unidad {unit} ---")
        process_unit(unit)
