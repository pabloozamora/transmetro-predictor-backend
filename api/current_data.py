import pandas as pd
from pathlib import Path
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional
import json

# === Utiliades de carga de datos ===

# Utilizar datos demo
DEMO_PATH = Path("D:\\2025\\UVG\\Tesis\\repos\\backend\\api\\db\\demo_data_best_day_copy.parquet")

# === Features necesarios para el modelo ===

FEATURES = [
    "Latitud","Longitud","LINEA","DIR","proxima_est_teorica",
    "dist_a_prox_m","dist_estacion_m","vel_mps","Altitud (m)","s_m","dist_m",
    "time_diff","dwell_same_xy_s","is_no_progress","progress_event",
    "hour","dow","is_weekend","is_peak"
]

def _is_peak(hour: int) -> int:
    return int((6 <= hour <= 9) or (16 <= hour <= 19))

def load_latest_data_per_unit(
    now: Optional[datetime] = None,
    simulate_live: bool = True,
) -> pd.DataFrame:
    """
    Carga el golden day y regresa el registro más reciente por unidad (Placa).
    - simulate_live=True: usa la hora/minuto de 'now' para cortar el día y simular un feed “hasta este momento”.
    - Ajusta hour/dow/is_weekend/is_peak a partir de 'now' para que el modelo sienta que es 'hoy'.
    """
    if now is None:
        # Ahora en UTC-6 (hora de Guatemala)
        now = datetime.now(timezone(timedelta(hours=-10)))
        
        # Restar horas para simular diferentes momentos del día
        now -= timedelta(hours=7)  # Ejemplo: simular 13 horas antes
        
    # Forzar una hora específica para pruebas
    now = now.replace(hour=14, minute=48, second=0, microsecond=0)

    df = pd.read_parquet(DEMO_PATH)
    # Asegura tipos mínimos
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha", "Placa"])

    # Normaliza strings (evitar problemas de tipo)
    for c in ["Placa","LINEA","DIR","proxima_est_teorica"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # --- Simulación de “tiempo actual” con el golden day ---
    if simulate_live:
        # tomamos la hora:minuto de 'now' y filtramos registros del golden day hasta esa hora:minuto
        hhmm = dtime(hour=now.hour, minute=now.minute, second=now.second)
        # Proyectamos al “mismo día” del golden day: comparamos solo por hora/minuto/segundo
        df = df[df["Fecha"].dt.time <= hhmm]

        # Si el filtro dejó vacío (por ejemplo muy temprano), relájalo a <= hora
        if df.empty:
            df = pd.read_parquet(DEMO_PATH)
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            df = df[df["Fecha"].dt.hour <= now.hour]

    # --- Último registro por unidad ---
    # Orden por Placa y tiempo, luego tomar el último por Placa
    df = df.sort_values(by=["Placa", "Fecha"])
    latest_per_unit = df.groupby("Placa", as_index=False).tail(1).copy()
    
    # Si la fecha del último registro es demasiado antigua (más de 2 horas), se puede filtrar
    # --- Filtrado por “reciente” usando SOLO hora del día simulada ---
    # Construir un "reloj actual" anclado al MISMO día que cada 'Fecha' del golden day
    sec_now = now.hour * 3600 + now.minute * 60 + now.second
    anchor_now = latest_per_unit["Fecha"].dt.normalize() + pd.to_timedelta(sec_now, unit="s")

    # Edad respecto a la hora actual simulada (en horas)
    age_hours = (anchor_now - latest_per_unit["Fecha"]).dt.total_seconds() / 3600.0

    # Conserva últimos pings con edad <= 1h (y descarta negativos por si hubiera “mirada hacia atrás” rara)
    latest_per_unit = latest_per_unit[(age_hours >= 0) & (age_hours <= 1.0)]

    # --- Ajuste de señales temporales para “hoy” ---
    latest_per_unit["hour"] = now.hour
    latest_per_unit["dow"] = now.weekday()       # Monday=0
    latest_per_unit["is_weekend"] = (latest_per_unit["dow"] >= 5).astype(int)
    latest_per_unit["is_peak"] = latest_per_unit["hour"].apply(_is_peak).astype(int)

    # --- Asegura columnas de FEATURES y tipos numéricos ---
    # Si falta alguna columna, crea con default razonable
    defaults = {
        "Latitud": 0.0,
        "Longitud": 0.0,
        "dwell_same_xy_s": 0.0,
        "Altitud (m)": 0.0,
        "dist_a_prox_m": 0.0,
        "dist_estacion_m": 0.0,
        "vel_mps": 0.0,
        "s_m": 0.0,
        "dist_m": 0.0,
        "time_diff": 60.0,
        "is_no_progress": 0,
        "progress_event": 1,
    }
    for c, v in defaults.items():
        if c not in latest_per_unit.columns:
            latest_per_unit[c] = v

    # Coerción numérica segura
    num_cols = [c for c in FEATURES if c not in ["LINEA","DIR","proxima_est_teorica"]]
    for c in num_cols:
        latest_per_unit[c] = pd.to_numeric(latest_per_unit[c], errors="coerce").fillna(defaults.get(c, 0.0))

    # Verificación rápida
    missing = [c for c in FEATURES if c not in latest_per_unit.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el payload: {missing}")

    # Orden final exacto
    payload_df = latest_per_unit[["Placa", "Fecha"] + FEATURES].copy()

    return payload_df

def get_latest_data_by_line_and_dir(line, _dir) -> dict:
    latest_data_df = load_latest_data_per_unit()
    
    next_data_by_station = {}
    
    # Agrupar por línea, dirección y estación
    grouped = latest_data_df.groupby(["LINEA", "DIR", "proxima_est_teorica"])
    
    # Devolver en diccionario sin ordenar: tomar el primer registro del grupo
    for (linea, dir_, estacion), group in grouped:
        if group.empty:
            continue
        if linea != line or dir_ != _dir:
            continue
        record_dict = group.iloc[0].to_dict()
        next_data_by_station[(linea, dir_, estacion)] = record_dict

    return next_data_by_station
    
