'''Script para construir datasets compactos por unidad,
filtrando filas innecesarias y optimizando tipos de datos.'''

from pathlib import Path
import pandas as pd, numpy as np, gc

pd.set_option("mode.copy_on_write", True)

OUT_DIR = Path("D:/2025/UVG/Tesis/repos/backend/compact_datasets_without_idle_rows")
OUT_DIR.mkdir(exist_ok=True)

# 1) Definir columnas necesarias
cat_cols = ["LINEA","DIR","proxima_est_teorica","DIR_init"]
num_cols = [
    "dist_a_prox_m","dist_estacion_m","Altitud (m)","s_m","dist_m","time_diff",
    "dwell_same_xy_s","is_no_progress","progress_event","ETA_proxima_est_s","Velocidad (km/h)"
]
meta_cols = ["Fecha","Placa","trip_id","block_id"]  # para ordenar/agrupación
keep_cols = cat_cols + num_cols + meta_cols

# Columnas finales para guardar (ya sin 'Velocidad (km/h)')
final_num = [
    "dist_a_prox_m","dist_estacion_m","vel_mps",
    "Altitud (m)","s_m","dist_m","time_diff","dwell_same_xy_s",
    "is_no_progress","progress_event"
]
final_time  = ["Fecha"]
final_group = ["Placa","trip_id","block_id"]
final_cols  = [c for c in (cat_cols + final_num + final_time + final_group + ["ETA_proxima_est_s"])]

# 2) Utilidades de downcast
def _downcast_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    float_cols = ["dist_a_prox_m","dist_estacion_m","Altitud (m)","s_m","dist_m","time_diff",
                  "dwell_same_xy_s","ETA_proxima_est_s","Velocidad (km/h)"]
    int_cols   = ["is_no_progress","progress_event","trip_id","block_id"]

    for c in float_cols:
        if c in chunk: chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
    for c in int_cols:
        if c in chunk: chunk[c] = pd.to_numeric(chunk[c], errors="coerce").fillna(0).astype("int32")

    for c in (set(cat_cols + ["Placa"]) & set(chunk.columns)):
        chunk[c] = chunk[c].astype("category")

    # Derivada esencial
    if "Velocidad (km/h)" in chunk:
        chunk["vel_mps"] = (chunk["Velocidad (km/h)"] / 3.6).astype("float32")

    return chunk

# 3) Filtro (compacto y barato)
def _mask_valid(chunk: pd.DataFrame) -> pd.Series:
    req = ["proxima_est_teorica","ETA_proxima_est_s","dist_a_prox_m","vel_mps","dwell_same_xy_s"]
    missing = [c for c in req if c not in chunk]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")
    
    eta   = pd.to_numeric(chunk["ETA_proxima_est_s"], errors="coerce")
    dist  = pd.to_numeric(chunk["dist_a_prox_m"],   errors="coerce")
    vel   = pd.to_numeric(chunk["vel_mps"],         errors="coerce")
    dwell = pd.to_numeric(chunk["dwell_same_xy_s"], errors="coerce")

    m = (
        chunk["proxima_est_teorica"].notna()
        & eta.notna()
        & (eta < 7200)
        & dist.between(0, 3000)
        & ( (vel > 0.3) | (dwell.fillna(np.inf) < 300) )
    )
    return m

# 4) Procesamiento por archivo (out-of-core)
def process_unit_csv(csv_path: Path, chunksize: int = 1_000_000, flush_rows: int = 5_000_000):
    print(f'--- Procesando {csv_path} ---')
    usecols_present = [c for c in keep_cols]  # si faltara alguna, read_csv la ignorará si no está
    df_iter = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols_present), parse_dates=["Fecha"], chunksize=chunksize)

    parts, acc_rows = [], 0
    for i, chunk in enumerate(df_iter):
        chunk = _downcast_chunk(chunk)

        mask = _mask_valid(chunk)
        # Seleccionar solo columnas finales disponibles
        cols_avail = [c for c in final_cols if c in chunk.columns]
        sub = chunk.loc[mask, cols_avail].copy()

        if len(sub):
            parts.append(sub)
            acc_rows += len(sub)

        # Liberar RAM del chunk original
        del chunk, sub

        # Volcar en partes para no acumular
        if acc_rows >= flush_rows:
            out_path = OUT_DIR / f"{csv_path.stem}_part_{i}.parquet"
            pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False, compression="zstd", engine="pyarrow")
            parts.clear()
            acc_rows = 0
            gc.collect()

    # Último volcado
    if parts:
        out_path = OUT_DIR / f"{csv_path.stem}_last.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False, compression="zstd", engine="pyarrow")
        parts.clear()
        gc.collect()
        
# --- Ejecución principal ---
if __name__ == "__main__":
    print('=== Iniciando construcción de datasets compactos ===')
    
    # Encontrar todas las unidades (carpetas en data with features)
    DATA_WITH_FEATURES_DIR = Path("D:/2025/UVG/Tesis/repos/backend/data_with_features")
    if not DATA_WITH_FEATURES_DIR.exists():
        print(f"No existe el directorio {DATA_WITH_FEATURES_DIR}. Fin.")
    units = [p.name for p in DATA_WITH_FEATURES_DIR.iterdir() if p.is_dir() and p.name != "maps"]
    
    if not units:
        print(f"No se encontraron unidades en {DATA_WITH_FEATURES_DIR}. Fin.")
        
    for unit in units:
        in_path = DATA_WITH_FEATURES_DIR / unit / f"{unit}_trips_with_next_station.csv"
        if not in_path.exists():
            print(f"No existe el archivo {in_path}. Omitiendo unidad {unit}.")
            continue
        
        process_unit_csv(in_path)
