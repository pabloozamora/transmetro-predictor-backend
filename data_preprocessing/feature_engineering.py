import numpy as np
import pandas as pd

R_EARTH = 6371000.0

# --------------------------
# Funciones auxiliares
# --------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2*R_EARTH*np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    """Rumbo (0-360) de punto1 -> punto2."""
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δλ = np.radians(lon2 - lon1)
    y = np.sin(Δλ) * np.cos(φ2)
    x = np.cos(φ1)*np.cos(φ2) - np.sin(φ1)*np.sin(φ2)*np.cos(Δλ)
    θ = np.degrees(np.arctan2(y, x))
    return (θ + 360) % 360

def ang_diff(a, b):
    """|a-b| en [0,180]."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def build_route_dicts(route_seq_df):
    """
    Devuelve:
      route_map[(linea,dir)] = [est1, est2, ...]  (orden)
      station_pos[(linea,dir,est)] = indice (0..N-1)
      next_after[(linea,dir,est)] = est_siguiente (o None si es terminal)
      neighbors[est] = set de (linea,dir) donde aparece
    """
    route_map = {}
    station_pos = {}
    next_after = {}
    neighbors = {}

    for (linea, d), g in route_seq_df.sort_values('ORDEN').groupby(['LINEA','DIR'], sort=False):
        seq = g['ESTACION'].tolist()
        route_map[(linea, d)] = seq
        for i, est in enumerate(seq):
            station_pos[(linea, d, est)] = i
            nxt = seq[i+1] if i+1 < len(seq) else None
            next_after[(linea, d, est)] = nxt
            neighbors.setdefault(est, set()).add((linea, d))
    return route_map, station_pos, next_after, neighbors

def _vector_dist_to_named_station(df, station_name_series, stations_df):
    lat_map = stations_df.set_index('ESTACION')['Latitud']
    lon_map = stations_df.set_index('ESTACION')['Longitud']
    tgt_lat = station_name_series.map(lat_map)
    tgt_lon = station_name_series.map(lon_map)
    return haversine_m(df['Latitud'].values, df['Longitud'].values,
                       tgt_lat.values, tgt_lon.values)
    
def infer_next_station_operational(
    df, stations_df, route_seq_df,
    heading_window_pts=5,       # puntos para rumbo suavizado
    min_speed_kmh_for_heading=3,
    lock_after_n_points=3,      # histéresis: puntos consecutivos coherentes para fijar dirección
    max_angle_for_candidate=90  # umbral angular para aceptar la dirección
):
    """
    Para cada trip, decide dirección (ida/vuelta) según heading y última estación vista,
    y calcula:
      - next_station_op      (próxima estación operativa)
      - dist_to_next_station_m_op
      - op_line/op_dir/op_dir_conf (diagnóstico)
    No usa futuro → apto como feature.
    """
    df = df.sort_values(['trip_id','Fecha']).copy()
    route_map, station_pos, next_after, neighbors = build_route_dicts(route_seq_df)

    # rumbo suave por trip (usando diferencias consecutivas)
    def smooth_heading(g):
        lat = g['Latitud'].to_numpy()
        lon = g['Longitud'].to_numpy()
        hdg = np.zeros(len(g))
        hdg[:] = np.nan
        for i in range(1, len(g)):
            hdg[i] = bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
        # media móvil simple sobre 'heading_window_pts'
        s = pd.Series(hdg)
        return s.rolling(heading_window_pts, min_periods=1).mean().to_numpy()
    
    df['heading_deg'] = df.groupby('trip_id', group_keys=False).apply(smooth_heading).reset_index(level=0, drop=True)

    df['next_station_op'] = np.nan
    df['dist_to_next_station_m_op'] = np.nan
    df['op_line'] = np.nan  # diagnóstico
    df['op_dir']  = np.nan  # diagnóstico
    df['op_dir_conf'] = 0.0

    for tid, g in df.groupby('trip_id', sort=False):
        g = g.copy()
        dir_locked = None         # (linea, dir) fijados por histéresis
        lock_counter = 0

        last_seen = g['estacion_cercana'].ffill()

        for i, row in g.iterrows():
            est_prev = last_seen.loc[i]
            if pd.isna(est_prev) or est_prev not in neighbors:
                # aún no tenemos estación previa → no podemos inferir próxima
                continue

            # candidatos: (linea, dir) que contienen a est_prev
            cands = list(neighbors[est_prev])

            # si ya hay dirección fijada e incluye est_prev, respetarla
            if dir_locked and est_prev in route_map[dir_locked]:
                chosen = dir_locked
            else:
                # calcular rumbo objetivo hacia el siguiente de cada candidato
                hdg_now = row['heading_deg']
                spd_now = row.get('Velocidad (km/h)', np.nan)
                angles = []
                for cand in cands: # cand = (Línea, dirección)
                    seq = route_map[cand]
                    j = station_pos.get((cand[0], cand[1], est_prev), None)
                    nxt = seq[j+1] if (j is not None and j+1 < len(seq)) else None
                    if nxt is None:
                        continue
                    # rumbo estación_prev -> estación_siguiente
                    s_prev = stations_df.set_index('ESTACION').loc[est_prev]
                    s_next = stations_df.set_index('ESTACION').loc[nxt]
                    hdg_target = bearing_deg(s_prev['Latitud'], s_prev['Longitud'],
                                             s_next['Latitud'], s_next['Longitud'])
                    if np.isnan(hdg_now):
                        ang = 180.0  # sin heading, no confiable
                    else:
                        ang = ang_diff(hdg_now, hdg_target)
                    angles.append((cand, nxt, ang))

                if not angles:
                    continue

                # elige el menor ángulo (si va razonablemente hacia esa dirección)
                cand, nxt_name, ang = min(angles, key=lambda x: x[2])
                conf = max(0.0, 1.0 - (ang / max_angle_for_candidate))  # 1→0 lineal
                # aplica histéresis si hay velocidad para confiar en el rumbo
                if (not np.isnan(spd_now) and spd_now >= min_speed_kmh_for_heading and ang <= max_angle_for_candidate):
                    if dir_locked == cand:
                        lock_counter += 1
                    else:
                        lock_counter = 1
                    if lock_counter >= lock_after_n_points:
                        dir_locked = cand
                chosen = cand

            # próxima estación y distancia operativa
            seq = route_map[chosen]
            j = station_pos[(chosen[0], chosen[1], est_prev)]
            nxt = seq[j+1] if j+1 < len(seq) else None
            if nxt is None:
                continue
            df.loc[i, 'next_station_op'] = nxt
            df.loc[i, 'op_line'] = chosen[0]
            df.loc[i, 'op_dir']  = chosen[1]
            # confianza opcional (0-1) basada en ángulo
            # recalcula ang para 'chosen'
            s_prev = stations_df.set_index('ESTACION').loc[est_prev]
            s_next = stations_df.set_index('ESTACION').loc[nxt]
            hdg_target = bearing_deg(s_prev['Latitud'], s_prev['Longitud'],
                                     s_next['Latitud'], s_next['Longitud'])
            ang = ang_diff(df.loc[i, 'heading_deg'], hdg_target) if not np.isnan(df.loc[i, 'heading_deg']) else 180.0
            df.loc[i, 'op_dir_conf'] = max(0.0, 1.0 - (ang / max_angle_for_candidate))

        # distancia a la próxima estación operativa (vectorizado dentro del trip)
        mask = df.index.isin(g.index)
        df.loc[mask, 'dist_to_next_station_m_op'] = _vector_dist_to_named_station(
            df.loc[mask], df.loc[mask, 'next_station_op'], stations_df
        )

    return df


# --------------------------
# Agregar features
# --------------------------

def add_features(
    df, stations_df, window_pts=10,
    include_hist_next_fields=False  # pon True si quieres dist_to_next_station_m histórica (ver nota)
):
    """
    df: limpio por trips (Fecha, Latitud, Longitud, Velocidad (km/h), estacion_cercana, dist_estacion_m, trip_id)
    stations_df: ['ESTACION','Latitud','Longitud']
    """
    df = df.sort_values(['trip_id','Fecha']).copy()

    # -----------------------------
    # Señales base y causales
    # -----------------------------
    df['gap_since_prev_ping_s'] = (
        df.groupby('trip_id')['Fecha'].diff().dt.total_seconds().clip(lower=0).fillna(0)
    )

    # Última estación "vista" (ffill causal)
    has_station = df['estacion_cercana'].notna()
    df['last_station_seen'] = df.groupby('trip_id')['estacion_cercana'].ffill()

    # Tiempo desde última estación "vista"
    t_last_seen = df['Fecha'].where(has_station).groupby(df['trip_id']).ffill()
    df['secs_since_last_station_seen'] = (df['Fecha'] - t_last_seen).dt.total_seconds()
    df.loc[t_last_seen.isna(), 'secs_since_last_station_seen'] = np.nan

    # Distancia actual a la última estación (vectorizado, sin apply)
    df['dist_to_last_station_now_m'] = _vector_dist_to_named_station(df, df['last_station_seen'], stations_df)

    # Tasa de aproximación (m/s) a la última estación (causal)
    d_dist = df.groupby('trip_id')['dist_to_last_station_now_m'].diff()
    dt = df['gap_since_prev_ping_s'].replace(0, np.nan)
    df['approach_rate_to_last_mps'] = (-d_dist / dt).replace([np.inf,-np.inf], np.nan).fillna(0)

    # Ventanas causales en puntos (no tiempo)
    df['speed_roll_mean'] = (
        df.groupby('trip_id')['Velocidad (km/h)'].apply(lambda s: s.rolling(window_pts, min_periods=1).mean())
    ).values
    df['speed_roll_std'] = (
        df.groupby('trip_id')['Velocidad (km/h)'].apply(lambda s: s.rolling(window_pts, min_periods=2).std())
    ).values

    # Min de distancia-a-estación más cercana en últimos N puntos (usando dist_estacion_m)
    roll_min = df.groupby('trip_id')['dist_estacion_m'].apply(lambda s: s.rolling(window_pts, min_periods=1).min())
    df['min_dist_lastN_m'] = roll_min.values

    # Tiempo desde ese mínimo (causal)
    df['secs_since_min_dist'] = (
        df.groupby('trip_id').apply(
            lambda g: (g['gap_since_prev_ping_s'] * (g['dist_estacion_m'] > g['min_dist_lastN_m']).astype(int)).cumsum()
        ).reset_index(level=0, drop=True)
    )

    return df

def add_label_eta_next_station(df):
    """
    Variable objetivo: segundos hasta la próxima estación (mirada hacia adelante).
    Nota: Usa info futura SOLO para construir el label.
    """
    df = df.sort_values(['trip_id','Fecha']).copy()
    next_t = df['Fecha'].where(df['estacion_cercana'].notna()).groupby(df['trip_id']).bfill()
    df['eta_to_next_station_s'] = (next_t - df['Fecha']).dt.total_seconds()
    return df


