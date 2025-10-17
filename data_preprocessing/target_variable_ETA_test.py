import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Optional, Tuple, Sequence
from math import radians, cos, sin, asin, sqrt

# ---------------------------
# Utilidades para "ETA ancla"
# ---------------------------

# Detectar bloques consecutivos donde la proxima_est_teorica no cambia
def _runs_of_equal(series: pd.Series) -> Iterable[Tuple[int, int, int]]:
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

# Determina el índice de llegada a la proxima_est_teorica para un run dado
def _arrival_index_for_run(
    i_run: int,
    runs: Sequence[Tuple[int,int,int]],
    dist_vals: np.ndarray,
    arrival_thresh: float,
    persist_n: int
) -> Tuple[Optional[int], str]:
    # Si hay run siguiente, toma el inicio del primer run futuro “estable” cuya longitud ≥ persist_n
    if i_run + 1 < len(runs):
        if persist_n > 0:
            target_rel = None
            for j in range(i_run+1, len(runs)):
                _, a_j, b_j = runs[j]
                if (b_j - a_j + 1) >= persist_n:
                    target_rel = a_j
                    break
            if target_rel is None:
                target_rel = runs[i_run+1][1]
            return target_rel, "change"
        else:
            return runs[i_run+1][1], "change"

    # Si es el último run: busca la primera muestra con dist_a_prox_m ≤ arrival_thresh
    _, a, b = runs[i_run]
    local = dist_vals[a:b+1]
    mask = (local <= arrival_thresh)
    if mask.any():
        return a + int(np.where(mask)[0][0]), "threshold"
    return None, "none"

# Por cada grupo (trip_id, block_id), fragmentado en runs de proxima_est_teorica constante,
# asigna ETA como tiempo hasta la llegada (cambio o umbral de distancia)
def _eta_anchor_for_group(
    g: pd.DataFrame,
    tcol: Optional[str],
    arrival_thresh: float,
    persist_n: int
) -> pd.DataFrame:
    n = len(g)
    out = pd.DataFrame(index=g.index, data={
        "ETA_proxima_est_s_anchor": np.full(n, np.nan, dtype=float),
        "arrival_source": np.array(["none"] * n, dtype=object),
    })
    if n == 0:
        return out

    prox = g["proxima_est_teorica"]
    dist = g["dist_a_prox_m"].astype(float).values
    
    # Fragmentar en runs de proxima_est_teorica constante
    runs = list(_runs_of_equal(prox))

    # Vector de tiempo (o índice si no hay tiempo)
    if tcol:
        t = g[tcol]
    else:
        t = pd.Series(np.arange(n, dtype=float), index=g.index)

    for i, (_, a, b) in enumerate(runs):
        
        # Obtener índice de llegada para este run
        arr_rel, src = _arrival_index_for_run(i, runs, dist, arrival_thresh, persist_n)
        if arr_rel is None:
            continue

        # Calcular ETA ancla para el bloque [a:b] como t(arrival) - t(a:b)
        if tcol:
            arrival_time = t.iloc[arr_rel]
            if pd.isna(arrival_time):
                continue
            eta_vals = (arrival_time - t.iloc[a:b+1]).dt.total_seconds().values
        else:
            arrival_time = t.iloc[arr_rel]
            eta_vals = (arrival_time - t.iloc[a:b+1].values).astype(float)

        eta_vals = np.where(eta_vals >= 0, eta_vals, np.nan)

        out.loc[g.index[a:b+1], "ETA_proxima_est_s_anchor"] = eta_vals
        out.loc[g.index[a:b+1], "arrival_source"] = src

    return out

# -------------------------------------
# Velocidad robusta y ETA cinemática
# -------------------------------------

def _compute_speed_for_group(
    g: pd.DataFrame,
    s_col: str,
    tcol: str,
    back_tolerance_m: float = 20.0,
    v_max_mps: float = 20.0,      # ~72 km/h (cap superior)
    roll_win: int = 7,            # ventana mediana
    ewm_halflife: float = 4.0     # suavizado exponencial
) -> pd.DataFrame:
    """
    Calcula velocidad instantánea ds/dt y versiones suavizadas:
      - speed_mps: instantánea robusta (clip y NaN si dt<=0)
      - speed_mps_med: mediana rodante
      - speed_mps_ewm: exponencial
      - speed_mps_eff: combinación (prefiere mediana, sino ewm)
    """
    out = pd.DataFrame(index=g.index)

    # Calcular velocidad instantánea (ds/dt)
    s = g[s_col].astype(float)
    dt = g[tcol].diff().dt.total_seconds()
    ds = s.diff()

    # Evitar retrocesos (ruido de proyección): tolera back_tolerance_m antes de anular
    ds_clean = ds.where((ds >= -back_tolerance_m), np.nan)
    ds_clean = ds_clean.clip(lower=0)  # no consumir distancia hacia atrás

    v = ds_clean / dt
    # Invalida si dt<=0 o v irreal
    v = v.where((dt > 0), np.nan)
    v = v.clip(upper=v_max_mps)

    # Suavizados
    
    # Mediana rodante
    v_med = v.rolling(roll_win, min_periods=max(2, roll_win//2)).median()
    v_ewm = v.ewm(halflife=ewm_halflife, adjust=False, ignore_na=True).mean()

    v_eff = v_med.combine_first(v_ewm)  # usa mediana si existe, sino ewm
    # Relleno suave hacia atrás para pequeños huecos
    v_eff = v_eff.fillna(method="ffill")

    out["speed_mps"] = v
    out["speed_mps_med"] = v_med
    out["speed_mps_ewm"] = v_ewm
    out["speed_mps_eff"] = v_eff
    return out

def _eta_kine_for_group(
    g: pd.DataFrame, # Bloque de datos (misma próxima estación)
    speed_cols: pd.DataFrame,
    dist_col: str,
    near_cap_radius_m: float = 120.0,
    v_cap_near_station_mps: float = 4.0,  # ~14.4 km/h
    v_min_operational_mps: float = 0.8,   # ~2.9 km/h (para evitar dividir por ~0)
    dwell_time_s: float = 15.0,           # tiempo de parada estimado
    min_valid_speed_pts: int = 3,
    win_valid_pts: int = 7,
    max_eta_s: float = 1800.0             # 30 min cap
) -> pd.Series:
    """
    ETA cinemática: dist_a_prox_m / v_eff (con ajustes realistas cerca de la estación)
      - Limita velocidad efectiva cerca de estación (cap) para modelar desaceleración
      - Añade dwell (tiempo de parada) para evitar ETA=0 al tocar andén
      - Rechaza casos con velocidad no confiable (pocos puntos válidos en ventana)
    """
    dist = g[dist_col].astype(float)
    v_eff = speed_cols["speed_mps_eff"].copy()

    # Cap de velocidad cuando estamos cerca de la estación
    near = dist <= near_cap_radius_m
    v_eff_adj = v_eff.where(~near, np.minimum(v_eff, v_cap_near_station_mps))

    # Reglas de confiabilidad de velocidad
    valid_mask = speed_cols["speed_mps"].notna()
    valid_cnt = valid_mask.rolling(win_valid_pts, min_periods=1).sum()
    speed_ok = (v_eff_adj >= v_min_operational_mps) & (valid_cnt >= min_valid_speed_pts)

    # ETA base
    eta = dist / v_eff_adj

    # Invalida cinemática si no confiable
    eta = pd.Series(eta, index=g.index)
    eta = eta.where(speed_ok, np.nan)
    
    # Para aquellos puntos que se quedaron sin ETA (primer registro del grupo, velocidad no confiable),
    # Rellenar con el siguiente válido hacia atrás (si existe) + la diferencia de tiempo entre estos registros
    
    g = g.sort_values("Fecha").copy()
    dates = pd.to_datetime(g["Fecha"])
    
    eta_filled = eta.copy()
    for i in range(len(eta)-2, -1, -1):
        if pd.isna(eta_filled.iat[i]) and not pd.isna(eta_filled.iat[i+1]):
            t_diff = (dates.iloc[i+1] - dates.iloc[i]).total_seconds()
            eta_filled.iat[i] = eta_filled.iat[i+1] + t_diff
                
    eta = eta_filled
    
    # Si el tiempo estimado es muy grande, sustituirlo por NaN
    eta = eta.where(eta <= max_eta_s, np.nan)
    
    return eta

# -------------------------------------
# Función principal (blend)
# -------------------------------------
def compute_eta_proxima_est_kinematic(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("trip_id","block_id"),
    time_col: str = "Fecha",
    s_col: str = "s_m",
    dist_col: str = "dist_a_prox_m",
    arrival_thresh: float = 20.0,
    persist_n: int = 0,
    # Velocidad
    back_tolerance_m: float = 20.0,
    v_max_mps: float = 20.0,
    roll_win: int = 7,
    ewm_halflife: float = 4.0,
    # Mezcla
    near_cap_radius_m: float = 120.0,
    v_cap_near_station_mps: float = 4.0,
    v_min_operational_mps: float = 0.8,
    dwell_time_s: float = 15.0,
    min_valid_speed_pts: int = 3,
    win_valid_pts: int = 7,
    max_eta_s: float = 1800.0,
) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      - speed_mps, speed_mps_med, speed_mps_ewm, speed_mps_eff
      - ETA_proxima_est_s_kine
      - ETA_proxima_est_s_anchor
      - ETA_proxima_est_s_final
      - ETA_proxima_est_min_final
      - arrival_source
    """
    df = df.copy()
    # Validación
    needed = ["proxima_est_teorica", dist_col, s_col, time_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Parseo tiempo y orden estable
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    real_groups = [c for c in group_cols if c in df.columns]
    sort_cols = real_groups + [time_col]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    
    # --- Partir por grupos de "proxima_est_teorica --- "
    # para evitar mezclar segmentos entre viajes, bloques o direcciones.
    part_cols = ["LINEA","DIR","DIR_init","trip_id","block_id","new_trip"]

    # Detecta cambio de partición (cambia cualquiera de estas columnas)
    part_change = df[part_cols].ne(df[part_cols].shift()).any(axis=1)

    # Detecta cambio de estación próxima (tratando NaN de forma estable)
    prox = df["proxima_est_teorica"].fillna("__NA__")
    station_change = prox.ne(prox.shift())
    
    # ID de segmento consecutivo: sube cuando hay cambio de estación o de partición
    df["segment_id"] = (part_change | station_change).cumsum()
    
    # Preasignar columnas de ancla en todo el DF
    df["ETA_proxima_est_s_anchor"] = np.nan
    df["arrival_source"] = "none"

    # 2) Calcular el ancla por partición completa (no por segmento)
    for _, gpart in df.groupby(part_cols, sort=False):
        anchor = _eta_anchor_for_group(
            gpart, tcol=time_col, arrival_thresh=arrival_thresh, persist_n=persist_n
        )
        # mismo índice -> asignación directa
        df.loc[gpart.index, ["ETA_proxima_est_s_anchor", "arrival_source"]] = \
            anchor[["ETA_proxima_est_s_anchor", "arrival_source"]].values

    # Procesar por segmento
    results = []
    for segment_id, g in df.groupby("segment_id"):
        # Velocidad
        sp = _compute_speed_for_group(
            g, s_col=s_col, tcol=time_col,
            back_tolerance_m=back_tolerance_m, v_max_mps=v_max_mps,
            roll_win=roll_win, ewm_halflife=ewm_halflife
        )

        # ETA cinemática
        eta_k = _eta_kine_for_group(
            g, sp, dist_col=dist_col,
            near_cap_radius_m=near_cap_radius_m,
            v_cap_near_station_mps=v_cap_near_station_mps,
            v_min_operational_mps=v_min_operational_mps,
            dwell_time_s=dwell_time_s,
            min_valid_speed_pts=min_valid_speed_pts,
            win_valid_pts=win_valid_pts,
            max_eta_s=max_eta_s
        )
        
        out = g.copy()
        out["ETA_proxima_est_s_kine"] = eta_k

        # usa el ancla precomputado en df (mismo índice)
        out["ETA_proxima_est_s_anchor"] = df.loc[g.index, "ETA_proxima_est_s_anchor"].values

        # Blend: el menor (llegada más temprana)
        eta_final = out["ETA_proxima_est_s_kine"].copy()
        eta_final.where(
            eta_final < out["ETA_proxima_est_s_anchor"],
            out["ETA_proxima_est_s_anchor"],
            inplace=True
        )
        out["ETA_proxima_est_s_final"] = eta_final
        out["ETA_proxima_est_min_final"] = eta_final / 60.0

        results.append(out)

    return pd.concat(results, ignore_index=True)


# ------------------------
# Uso como script
# ------------------------
if __name__ == "__main__":
    # Ejemplo de uso
    UNIT = "u096"
    in_path = Path(f"D:/2025/UVG/Tesis/repos/backend/data_with_features/{UNIT}/{UNIT}_trips_with_next_station.csv")
    out_path = in_path.with_name(f"{UNIT}_trips_with_eta_kinematic.csv")

    df = pd.read_csv(in_path)
    
    # Probar con un solo trip
    df = df[df["trip_id"] == 2].copy()

    df_eta = compute_eta_proxima_est_kinematic(
        df,
        group_cols=("trip_id","block_id"),
        time_col="Fecha",
        s_col="dist_a_prox_m",   # usar dist_a_prox_m como s_m (distancia recorrida)
        dist_col="dist_a_prox_m",
        arrival_thresh=20.0,      # llegada si último run y dist<=20m
        persist_n=2,              # histeresis de cambio (opcional)
        # velocidad
        back_tolerance_m=20.0,
        v_max_mps=20.0,
        roll_win=7,
        ewm_halflife=4.0,
        # mezcla y realismo cerca de estación
        near_cap_radius_m=120.0,
        v_cap_near_station_mps=4.0,
        v_min_operational_mps=0.8,
        dwell_time_s=15.0,
        min_valid_speed_pts=3,
        win_valid_pts=7,
        max_eta_s=3600.0,
    )

    df_eta.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")
