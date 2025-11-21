'''Script para verificar que los archivos Parquet generados coincidan en número de filas
con el filtrado esperado desde los CSVs originales.'''

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from pathlib import Path

OUT_DIR = Path("../compact_datasets")

# Debe replicar EXACTO el downcast + mask al construir los Parquet
cat_cols = ["LINEA","DIR","proxima_est_teorica","DIR_init"]
num_cols = [
    "dist_a_prox_m","dist_estacion_m","Altitud (m)","s_m","dist_m","time_diff",
    "dwell_same_xy_s","is_no_progress","progress_event","ETA_proxima_est_s","Velocidad (km/h)"
]
meta_cols = ["Fecha","Placa","trip_id","block_id"]
keep_cols = cat_cols + num_cols + meta_cols

def expected_count_from_csv(csv_path: Path, chunksize=1_000_000) -> int:
    exp_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=lambda c: c in set(keep_cols),
                             parse_dates=["Fecha"], chunksize=chunksize):
        # Downcast mínimo necesario para la máscara
        for c in ["dist_a_prox_m","dist_estacion_m","Altitud (m)","s_m","dist_m","time_diff",
                  "dwell_same_xy_s","ETA_proxima_est_s","Velocidad (km/h)"]:
            if c in chunk: chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype("float32")
        if "is_no_progress" in chunk: chunk["is_no_progress"] = pd.to_numeric(chunk["is_no_progress"], errors="coerce").fillna(0).astype("int32")
        if "progress_event" in chunk: chunk["progress_event"] = pd.to_numeric(chunk["progress_event"], errors="coerce").fillna(0).astype("int32")

        chunk["vel_mps"] = (chunk["Velocidad (km/h)"] / 3.6).astype("float32") if "Velocidad (km/h)" in chunk else 0.0

        mask = (
            chunk.get("proxima_est_teorica").notna()
            & chunk.get("ETA_proxima_est_s").notna()
            & chunk.get("dist_a_prox_m").between(0, 3000)
            & ((chunk.get("vel_mps") > 0.3) | (chunk.get("dwell_same_xy_s").fillna(0) < 300))
        )
        exp_rows += int(mask.sum())
    return exp_rows

def actual_count_from_parquets(stem: str, out_dir=OUT_DIR) -> int:
    # Cuenta filas de todos los parquet cuyo nombre empiece con <stem> (part_*, last.parquet)
    files = list(Path(out_dir).glob(f"{stem}_*.parquet"))
    if not files:
        # también intenta el caso de un único parquet sin sufijo
        files = list(Path(out_dir).glob(f"{stem}.parquet"))
    if not files:
        return 0
    d = ds.dataset([str(f) for f in files], format="parquet")
    return d.count_rows()

def check_unit(unit):
    csv_path = Path(f"../data_with_features/{unit}/{unit}_trips_with_next_station.csv")
    stem = csv_path.stem  # "056_trips_with_next_station"
    exp = expected_count_from_csv(csv_path)
    act = actual_count_from_parquets(stem)
    print(f"CSV esperado (filtrado) = {exp:,}  vs  Parquet guardado = {act:,}")
    assert exp == act, "❌ Mismatch de filas; revisar proceso."
    print("✅ Coinciden las filas filtradas con lo guardado en Parquet.")
    
if __name__ == "__main__":
    print("=== Verificando archivos Parquet contra CSVs originales ===")
    DATA_WITH_FEATURES_DIR = Path("../data_with_features")
    if not DATA_WITH_FEATURES_DIR.exists():
        print(f"No existe el directorio {DATA_WITH_FEATURES_DIR}. Fin.")
    units = [p.name for p in DATA_WITH_FEATURES_DIR.iterdir() if p.is_dir() and p.name != "maps"]
    
    if not units:
        print(f"No se encontraron unidades en {DATA_WITH_FEATURES_DIR}. Fin.")
        
    for unit in units:
        print(f"--- Verificando unidad {unit} ---")
        try:
            check_unit(unit)
        except AssertionError as e:
            print(str(e))
