from pathlib import Path
import pandas as pd, numpy as np, gc

COMPACT_DIR = Path("D:/2025/UVG/Tesis/repos/backend/compact_datasets_without_idle_rows")
FEATS_DIR = Path("D:/2025/UVG/Tesis/repos/backend/features_ready_without_idle_rows")
FEATS_DIR.mkdir(exist_ok=True)

# Columnas de entrada (ya definidas en compact datasets)
cat_cols  = ["LINEA","DIR","proxima_est_teorica","DIR_init"]
base_cols = [
    "Fecha","Placa","trip_id","block_id",
    "dist_a_prox_m","dist_estacion_m","vel_mps",
    "Altitud (m)","s_m","dist_m","time_diff","dwell_same_xy_s",
    "is_no_progress","progress_event","ETA_proxima_est_s"
]
in_cols = list(dict.fromkeys(cat_cols + base_cols))  # sin duplicados, respeta orden

# Features nuevas a crear
new_time_cols = ["hour","dow","is_weekend","is_peak"]

def process_one_stem(stem: str):
    # Junta todas las partes de ese origen (para no cortar trips)
    files = sorted(COMPACT_DIR.glob(f"{stem}_*.parquet")) or [COMPACT_DIR / f"{stem}.parquet"]
    if not files[0].exists():
        print(f"⚠️ sin archivos para {stem}")
        return

    # Carga solo columnas necesarias
    dfs = [pd.read_parquet(f, columns=[c for c in in_cols if c in pd.read_parquet(f).columns]) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    # Tipos livianos y categóricas
    for c in cat_cols:
        if c in df: df[c] = df[c].astype("category")
    for c in ["is_no_progress","progress_event","trip_id","block_id"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    for c in ["dist_a_prox_m","dist_estacion_m","vel_mps","Altitud (m)","s_m","dist_m","time_diff","dwell_same_xy_s","ETA_proxima_est_s"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    # Features temporales
    df["hour"] = df["Fecha"].dt.hour.astype("int8")
    df["dow"] = df["Fecha"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["is_peak"] = (df["hour"].between(6, 9) | df["hour"].between(16, 19)).astype("int8")

    # Selección final de columnas a guardar
    out_cols = [c for c in (cat_cols + base_cols + new_time_cols) if c in df.columns]
    data = df[out_cols]
    del df
    gc.collect()

    out_path = FEATS_DIR / f"{stem}_features.parquet"
    data.to_parquet(out_path, index=False, compression="zstd", engine="pyarrow")
    del data
    gc.collect()
    print(f"✅ {stem}: features → {out_path}")

# Ejecuta para todos los “stems” presentes en compact_datasets
stems = sorted({p.stem.split("_part_")[0].split("_last")[0] for p in COMPACT_DIR.glob("*.parquet")})
for stem in stems:
    process_one_stem(stem)
