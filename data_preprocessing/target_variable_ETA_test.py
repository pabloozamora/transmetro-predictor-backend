# eta_proxima_est.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Tuple

def _runs_of_equal(series: pd.Series) -> Iterable[Tuple[int, int, int]]:
    """
    Devuelve runs de valores iguales sobre la serie (ya ordenada), como tuplas (run_id, start_idx_rel, end_idx_rel).
    Los índices devueltos son relativos al inicio de la serie (0..n-1).
    """
    vals = series.astype("string").fillna("<NA>").values
    n = len(vals)
    if n == 0:
        return []
    run_start = np.zeros(n, dtype=bool)
    run_start[0] = True
    run_start[1:] = (vals[1:] != vals[:-1])
    run_id = np.cumsum(run_start) - 1
    out = []
    for rid in np.unique(run_id):
        idxs = np.where(run_id == rid)[0]
        out.append((int(rid), int(idxs[0]), int(idxs[-1])))
    return out


def _arrival_index_for_run(
    i_run: int,
    runs: Iterable[Tuple[int,int,int]],
    dist_vals: np.ndarray,
    arrival_thresh: float,
    persist_n: int
) -> Tuple[Optional[int], str]:
    """
    Determina el índice (relativo al grupo) donde se considera que se llega a la estación del run i_run.
    Estrategia:
      1) Si existe un run futuro con la siguiente estación, se usa el inicio del primer run "estable"
         (longitud >= persist_n si persist_n>0; si persist_n==0, el inmediato).
      2) Si es el último run, se usa el primer índice dentro del run con dist <= arrival_thresh.
    Retorna (arrival_idx_rel, arrival_source).
    """
    # Caso con siguiente run: llegar cuando cambia la proxima_est_teorica
    if i_run + 1 < len(runs):
        
        # Ver si dentro del run actual ya se llegó a la estación (umbral de distancia)
        rid, a, b = runs[i_run]
        local = dist_vals[a:b+1]
        mask = (local <= arrival_thresh)
        if mask.any():
            return a + int(np.where(mask)[0][0]), "threshold"
        
        if persist_n > 0:
            # Buscar el primer run futuro con longitud >= persist_n
            target_rel = None
            for j in range(i_run+1, len(runs)):
                _, a_j, b_j = runs[j]
                if (b_j - a_j + 1) >= persist_n:
                    target_rel = a_j  # inicio del run estable
                    break
            # Si no hay ninguno estable, usa el inmediato
            if target_rel is None:
                target_rel = runs[i_run+1][1]
            return target_rel, "change"
        else:
            return runs[i_run+1][1], "change"

    # Último run: usar umbral de distancia
    rid, a, b = runs[i_run]
    local = dist_vals[a:b+1]
    mask = (local <= arrival_thresh)
    if mask.any():
        return a + int(np.where(mask)[0][0]), "threshold"
    return None, "none"


def _compute_eta_for_group(
    g: pd.DataFrame,
    arrival_thresh: float,
    persist_n: int
) -> pd.DataFrame:
    """
    Calcula ETA_proxima_est para un grupo (un trip_id y opcionalmente block_id).
    Devuelve DataFrame con mismas filas/índices que g y columnas:
      - ETA_proxima_est_s
      - arrival_source
    """
    n = len(g)
    out = pd.DataFrame(index=g.index, data={
        "ETA_proxima_est_s": np.full(n, np.nan, dtype=float),
        "arrival_source": np.array(["none"] * n, dtype=object),
    })

    if n == 0:
        return out

    prox = g["proxima_est_teorica"]
    dist = g["dist_a_prox_m"].astype(float).values
    runs = list(_runs_of_equal(prox))

    # Vector de tiempo
    t = g['Fecha']

    for i, (_, a, b) in enumerate(runs):
        arr_rel, src = _arrival_index_for_run(i, runs, dist, arrival_thresh, persist_n)
        if arr_rel is None:
            continue

        # ETA = t(arrival) - t(k), para k in [a..b]
        arrival_time = t.iloc[arr_rel]
        # Si hay NaT, no asignamos
        if pd.isna(arrival_time):
            continue
        eta_vals = (arrival_time - t.iloc[a:b+1]).dt.total_seconds().values

        # Evitar valores negativos si hubiera desorden temporal
        eta_vals = np.where(eta_vals >= 0, eta_vals, np.nan)
        
        # Advertir de ETAs excesivamente grandes (>6 horas)
        if np.any(eta_vals > 21600):
            print(f"Advertencia: ETAs excesivamente grandes (>6 horas) en el trip {g['trip_id'].iloc[a]} - bloque {g['block_id'].iloc[a]}")

        out.loc[g.index[a:b+1], "ETA_proxima_est_s"] = eta_vals
        out.loc[g.index[a:b+1], "arrival_source"] = src

    return out


def compute_eta_proxima_est(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("trip_id","block_id"),
    arrival_thresh: float = 20.0,
    persist_n: int = 0,
    sort_within_groups: bool = True
) -> pd.DataFrame:
    """
    Calcula ETA_proxima_est y añade:
      - ETA_proxima_est_s (segundos; NaN si no hay tiempo)
      - ETA_proxima_est_min
      - eta_available (True si ETA válido y >=0)
      - arrival_source ("change" | "threshold" | "none")

    Parámetros:
      - group_cols: columnas para agrupar  p.ej. ("block_id", "trip_id")
      - arrival_thresh: umbral en metros para “llegó” si es el último run del grupo.
      - persist_n: histeresis en muestras para considerar válido el cambio de estación (0 = desactivado).
      - sort_within_groups: si True, ordena por 'Fecha' dentro del grupo para garantizar monotonicidad temporal.
    """
    df = df.copy()

    # Validar columnas
    for c in ("proxima_est_teorica","dist_a_prox_m"):
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida: '{c}'")

    # Determinar/parsear columna de tiempo
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    # Orden global estable para que los grupos mantengan coherencia visual,
    # pero el cálculo real se hará por grupo:
    df = df.sort_values(list(group_cols) + ['Fecha'], kind="mergesort").reset_index(drop=True)

    # Aplica por grupo
    pieces = []
    for _, g in df.groupby(list(group_cols), sort=False):
        g_local = g.sort_values('Fecha', kind="mergesort") if sort_within_groups else g
        res = _compute_eta_for_group(g_local, arrival_thresh, persist_n)
        # Adjunta resultados al grupo original (mismas filas/índices)
        g_local = g_local.copy()
        g_local["ETA_proxima_est_s"] = res["ETA_proxima_est_s"].astype(float)
        g_local["arrival_source"] = res["arrival_source"]
        pieces.append(g_local)

    out = pd.concat(pieces, axis=0).sort_index()
    out["ETA_proxima_est_min"] = out["ETA_proxima_est_s"] / 60.0
    out["eta_available"] = np.isfinite(out["ETA_proxima_est_s"]) & (out["ETA_proxima_est_s"] >= 0)
    return out

def process_unit(unit):
    UNIT = unit
    # Cambia estas rutas según tus archivos
    in_path = f'D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{UNIT}\\{UNIT}_trips_with_next_station.csv'
    out_path = in_path

    df = pd.read_csv(in_path)

    df_eta = compute_eta_proxima_est(
        df,
        group_cols=("trip_id","block_id"),
        arrival_thresh=20.0,                # metros
        persist_n=0,                        # sin histeresis
        sort_within_groups=True
    )

    df_eta.to_csv(out_path, index=False)
    print(f"Archivo guardado en: {out_path}")
    
# --- Ejecución principal ---
if __name__ == "__main__":
    print('=== Iniciando cálculo de variable objetivo (ETA) ===')
    
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
