"""
Pipeline de limpieza de trayectorias Transmetro (multi-unidad)

Uso:
  python pipeline_transmetro_clean.py --input-dir ./joined_data --stations ./data/Estaciones.xlsx --out ./clean_data --all --maps --summary
  python pipeline_transmetro_clean.py --input-dir ./joined_data --stations ./data/Estaciones.xlsx --out ./clean_data --units u001 u049 --pre-roll 8 --post-roll 2 --pause-threshold 600 --threshold-m 100
"""

import argparse, os, sys, glob, math
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros/columnas
# -----------------------------
COL_FECHA = 'Fecha'
COL_TRIP  = 'trip_id'
COL_EST   = 'estacion_cercana'
COL_VEL   = 'Velocidad (km/h)'
COL_LAT   = 'Latitud'
COL_LON   = 'Longitud'

DEFAULT_THRESHOLD_M = 100        # Distancia máxima para asignar estación más cercana
DEFAULT_PAUSE_S     = 600        # Pausa (seg) para cortar viajes
DEFAULT_PRE_MIN     = 8          # Minutos antes de 1ra estación
DEFAULT_POST_MIN    = 2          # Minutos después de última estación
DEFAULT_USE_IDLE_TRIM = False    # Utilizar velocidad 0 km/h para recortar trayectorias
DEFAULT_V0_KMH      = 1.0
DEFAULT_MAX_IDLE_S  = 300

# -----------------------------
# Cargar estaciones + BallTree
# -----------------------------
def load_stations_and_tree(stations_xlsx):
    from sklearn.neighbors import BallTree
    st = pd.read_excel(stations_xlsx)

    # Normalizar columnas esperadas
    if 'POSICIÓN' in st.columns and ('Latitud' not in st or 'Longitud' not in st):
        coords = st['POSICIÓN'].str.split(',', expand=True)
        st['Latitud']  = coords[0].astype(float)
        st['Longitud'] = coords[1].astype(float)

    st['lat_rad'] = np.deg2rad(st['Latitud'])
    st['lon_rad'] = np.deg2rad(st['Longitud'])
    station_coords = np.vstack([st['lat_rad'], st['lon_rad']]).T
    tree = BallTree(station_coords, metric='haversine')
    return st, tree

# -----------------------------
# Lectura CSV de unidad
# -----------------------------
def load_unit_csv(path):
    dtypes = {
        COL_LAT: 'float32',
        COL_LON: 'float32',
        COL_VEL: 'float16',
        'Placa': 'string'
    }
    df = pd.read_csv(path, header=0, dtype=dtypes, low_memory=False)
    # Normaliza fecha
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')
    # Limpieza básica
    for col in ['Hora', 'Alias', 'Fecha de registro', 'Hora de registro', 'Georeferencia']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df.drop_duplicates(subset=[COL_FECHA, COL_LAT, COL_LON], inplace=True)
    df = df.sort_values(COL_FECHA).reset_index(drop=True)
    return df

# Haversine vectorizado (metros)
def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def add_step_metrics(df, max_speed_kmh=90, max_step_m=500, max_gap_s=900):
    """Calcula distancia/tiempo/velocidad entre puntos consecutivos y marca 'jump'."""
    if df.empty:
        df = df.copy()
        df['step_dt_s'] = df['step_dist_m'] = df['step_speed_kmh'] = 0.0
        df['jump'] = False
        return df

    d = df.sort_values(COL_FECHA).copy()
    dt = d[COL_FECHA].diff().dt.total_seconds().fillna(0).to_numpy()
    dist = np.zeros(len(d), dtype=float)
    dist[1:] = _haversine_m(d[COL_LAT].to_numpy()[:-1], d[COL_LON].to_numpy()[:-1],
                            d[COL_LAT].to_numpy()[1:],  d[COL_LON].to_numpy()[1:])
    speed = np.where(dt > 0, (dist/dt)*3.6, 0.0)  # km/h

    d['step_dt_s'] = dt
    d['step_dist_m'] = dist
    d['step_speed_kmh'] = speed

    # “salto” si velocidad imposible, distancia gigante o hueco de tiempo grande
    d['jump'] = (d['step_speed_kmh'] > max_speed_kmh) | (d['step_dist_m'] > max_step_m) | (d['step_dt_s'] > max_gap_s)
    # el primer punto no tiene “paso previo”: fuerza False
    d.loc[d.index[0], 'jump'] = False
    return d

# -----------------------------
# Asignación de cada registro a estación más cercana (k=1) + umbral
# -----------------------------
def assign_nearest_station(df, stations_df, tree, threshold_m=DEFAULT_THRESHOLD_M):
    df['lat_rad'] = np.deg2rad(df[COL_LAT].astype(float))
    df['lon_rad'] = np.deg2rad(df[COL_LON].astype(float))
    gps = np.vstack([df['lat_rad'], df['lon_rad']]).T
    dist_rad, idx = tree.query(gps, k=1)
    df['dist_estacion_m'] = dist_rad[:, 0] * 6_371_000
    df['estacion_idx']    = idx[:, 0]
    # Nombre de estación
    if 'ESTACION' in stations_df.columns:
        df[COL_EST] = stations_df['ESTACION'].iloc[df['estacion_idx']].values
    else:
        # Fallback a alguna columna de nombre
        name_col = [c for c in stations_df.columns if c.lower().startswith('est')][0]
        df[COL_EST] = stations_df[name_col].iloc[df['estacion_idx']].values
    # Umbral (asignación confiable)
    df.loc[df['dist_estacion_m'] > threshold_m, COL_EST] = None
    return df

# -----------------------------
# Filtrar días sin estaciones
# -----------------------------
def filter_days_without_stations(df):
    df['Dia'] = df[COL_FECHA].dt.date
    agg = df.groupby('Dia')[COL_EST].nunique().reset_index(name='num_estaciones')
    days_without_stations = agg.loc[agg['num_estaciones'] == 0, 'Dia']
    df_filtrado = df[~df[COL_FECHA].dt.date.isin(days_without_stations)].copy()
    return df_filtrado, agg, days_without_stations

# -----------------------------
# Segmentación en trips por pausas
# -----------------------------
def segment_trips(df, pause_s=DEFAULT_PAUSE_S):
    df = df.sort_values(COL_FECHA).copy()
    df['time_diff'] = df[COL_FECHA].diff().dt.total_seconds().fillna(0)
    day_change = df[COL_FECHA].dt.date != df[COL_FECHA].shift().dt.date
    df['new_trip'] = ((df['time_diff'] > pause_s) | day_change).astype(int)
    df[COL_TRIP] = df['new_trip'].cumsum()
    return df

# -----------------------------
# Recorte por estaciones con pre/post roll
# -----------------------------
def trim_by_stations(df_trip, pre_min=DEFAULT_PRE_MIN, post_min=DEFAULT_POST_MIN):
    df_trip = df_trip.sort_values(COL_FECHA).copy()
    mask = df_trip[COL_EST].notna()
    if not mask.any():
        return df_trip.iloc[0:0].copy()
    first_idx = df_trip.index[mask][0]
    last_idx  = df_trip.index[mask][-1]
    t_first = df_trip.loc[first_idx, COL_FECHA]
    t_last  = df_trip.loc[last_idx,  COL_FECHA]
    t0 = t_first - pd.Timedelta(minutes=pre_min)
    t1 = t_last  + pd.Timedelta(minutes=post_min)
    return df_trip[(df_trip[COL_FECHA] >= t0) & (df_trip[COL_FECHA] <= t1)].copy()

# -----------------------------
# Recorte de colas estacionadas (opcional)
# -----------------------------
def trim_by_idle_queues(df_trip, v0_kmh=DEFAULT_V0_KMH, max_idle_s=DEFAULT_MAX_IDLE_S):
    if df_trip.empty or COL_VEL not in df_trip.columns:
        return df_trip
    d = df_trip.sort_values(COL_FECHA).copy()
    dt = d[COL_FECHA].diff().dt.total_seconds().fillna(0).to_numpy()
    v  = d[COL_VEL].astype(float).to_numpy()
    idx = d.index.to_numpy()

    # Inicial
    idle = 0.0
    start_idx = idx[0]
    for i in range(len(d)):
        if v[i] <= v0_kmh:
            idle += dt[i]
        else:
            if idle >= max_idle_s:
                start_idx = idx[i]
            break
    # Final
    idle_tail = 0.0
    end_idx = idx[-1]
    rev_dt = np.r_[0, np.diff(d[COL_FECHA].iloc[::-1]).astype('timedelta64[s]').astype(float)]
    v_rev = v[::-1]
    idx_rev = idx[::-1]
    for i in range(len(d)):
        if v_rev[i] <= v0_kmh:
            idle_tail += rev_dt[i]
        else:
            if idle_tail >= max_idle_s:
                end_idx = idx_rev[i]
            break
    return d.loc[start_idx:end_idx].copy()

# -----------------------------
# Limpieza de trips
# -----------------------------
def clean_trips_by_stations(df_filtrado, pre_min, post_min, use_idle_trim, v0_kmh, max_idle_s):
    assert {COL_FECHA, COL_EST}.issubset(df_filtrado.columns)

    df = df_filtrado.copy()
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors='coerce')
    df = df.sort_values([COL_TRIP, COL_FECHA])

    # Métricas previas (diagnóstico)
    prev = (
        df.groupby(COL_TRIP)
          .agg(
              start=(COL_FECHA,'first'),
              end=(COL_FECHA,'last'),
              puntos=(COL_FECHA,'count'),
              est_uniq=(COL_EST, lambda s: s.dropna().nunique()),
              dur_s=(COL_FECHA, lambda s: (s.max()-s.min()).total_seconds() if not s.empty else 0.0)
          ).reset_index()
    )

    # Filtrar trips “válidos” (antes del recorte)
    prev['dur_s'] = pd.to_numeric(prev['dur_s'], errors='coerce').fillna(0)
    valid_ids = set(prev[(prev['dur_s'] >= 300) & (prev['est_uniq'] >= 2)][COL_TRIP])

    trims = []
    for trip_id, g in df.groupby(COL_TRIP, sort=False):
        if trip_id not in valid_ids:
            continue
        g1 = trim_by_stations(g, pre_min=pre_min, post_min=post_min)
        if g1.empty:
            continue
        g2 = trim_by_idle_queues(g1, v0_kmh=v0_kmh, max_idle_s=max_idle_s) if use_idle_trim else g1
        # Barrera post
        dur = (g2[COL_FECHA].max() - g2[COL_FECHA].min()).total_seconds() if not g2.empty else 0
        if (g2[COL_EST].dropna().nunique() < 2) or (dur < 300) or (len(g2) < 5):
            continue
        trims.append(g2)

    clean_df = pd.concat(trims, ignore_index=False) if trims else df.iloc[0:0].copy()
    clean_df = clean_df.sort_values([COL_TRIP, COL_FECHA]).copy()

    # Métricas posteriores
    post = (
        clean_df.groupby(COL_TRIP)
                 .agg(
                     start=(COL_FECHA,'first'),
                     end=(COL_FECHA,'last'),
                     puntos=(COL_FECHA,'count'),
                     est_uniq=(COL_EST, lambda s: s.dropna().nunique()),
                     dur_s=(COL_FECHA, lambda s: (s.max()-s.min()).total_seconds())
                 ).reset_index()
    )

    # Drops
    drops = prev[[COL_TRIP,'puntos']].merge(
        post[[COL_TRIP,'puntos']].rename(columns={'puntos':'puntos_post'}),
        on=COL_TRIP, how='left'
    )
    drops['puntos_post'] = drops['puntos_post'].fillna(0).astype(int)
    drops['puntos_eliminados'] = drops['puntos'] - drops['puntos_post']

    return clean_df, post, drops, prev

# -----------------------------
# Guardar resultados
# -----------------------------
def save_outputs(unit, out_dir, clean_df, summary_post, drops, summary_pre):
    unit_dir = os.path.join(out_dir, f"{unit}")
    os.makedirs(unit_dir, exist_ok=True)

    clean_df.to_csv(os.path.join(unit_dir, f"{unit}_clean_trips.csv"), index=False)
    summary_post.to_csv(os.path.join(unit_dir, f"{unit}_post_trips_summary.csv"), index=False)
    drops.to_csv(os.path.join(unit_dir, f"{unit}_drops_per_trip.csv"), index=False)
    summary_pre.to_csv(os.path.join(unit_dir, f"{unit}_pre_trips_summary.csv"), index=False)

# -----------------------------
# Guardar mapa de cada viaje de la unidad
# -----------------------------
def export_trips_maps(unit, out_dir, clean_df, no_station_days_map_path):
    try:
        import folium
        from folium.plugins import MarkerCluster
    except Exception as e:
        print(f"[{unit}] folium no disponible, se omiten mapas. {e}")
        return
    maps_dir = os.path.join(out_dir, "maps", f"{unit}")
    os.makedirs(maps_dir, exist_ok=True)

    def make_trip_map(df_trip, trip_id):
        if df_trip.empty: return None
        lat_center = df_trip[COL_LAT].mean()
        lon_center = df_trip[COL_LON].mean()
        m = folium.Map(location=[lat_center, lon_center], zoom_start=13, control_scale=True)
        df_trip = df_trip.sort_values(COL_FECHA)
        coords = df_trip[[COL_LAT, COL_LON]].values.tolist()
        folium.PolyLine(coords, weight=4, opacity=0.8).add_to(m)
        start = df_trip.iloc[0]; end = df_trip.iloc[-1]
        folium.Marker([start[COL_LAT], start[COL_LON]], popup=f"Inicio: {start[COL_FECHA]}", tooltip="Inicio").add_to(m)
        folium.Marker([end[COL_LAT],   end[COL_LON]],   popup=f"Fin: {end[COL_FECHA]}",   tooltip="Fin",
                      icon=folium.Icon(icon="flag")).add_to(m)
        if COL_EST in df_trip.columns:
            st_mask = df_trip[COL_EST].notna()
            if st_mask.any():
                fg = folium.FeatureGroup(name="Puntos con estación").add_to(m)
                for _, r in df_trip[st_mask].iterrows():
                    folium.CircleMarker(
                        [float(r[COL_LAT]), float(r[COL_LON])],
                        radius=3,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"{r[COL_FECHA]} - {r[COL_EST]}"
                    ).add_to(fg)
        out_path = os.path.join(maps_dir, f"trip_{trip_id}.html")
        m.save(out_path)
        return out_path

    links = []
    for tid, g in clean_df.groupby(COL_TRIP, sort=True):
        p = make_trip_map(g, tid)
        if p: links.append((tid, p))

    # index
    index_path = os.path.join(maps_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(f"<h2>Mapas por viaje ({unit})</h2><ul>")
        for tid, path in links:
            fname = os.path.basename(path)
            f.write(f'<li><a href="{fname}" target="_blank">Trip {tid}</a></li>')
        if no_station_days_map_path:
            fname = os.path.basename(no_station_days_map_path)
            f.write(f'<li><a href="{fname}" target="_blank">Días sin estaciones</a></li>')
        f.write("</ul>")
    print(f"[{unit}] Mapas listos en: {maps_dir}")
    
# -----------------------------
# Guardar histogramas
# -----------------------------
def export_histograms(unit, out_dir, df_original, agg_est):
    """
    Guarda dos PNGs por unidad:
      - Histograma de velocidades
      - Histograma de # de estaciones por día
    """
    import numpy as np
    unit_dir = os.path.join(out_dir, f"{unit}")
    os.makedirs(unit_dir, exist_ok=True)

    # --- Histograma de velocidades (usa el df original de la unidad) ---
    if 'Velocidad (km/h)' in df_original.columns and df_original['Velocidad (km/h)'].notna().any():
        vel = df_original.get('Velocidad (km/h)')
        if vel is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(vel.dropna(), bins=60, alpha=0.8)
            ax.set_title(f"Histograma de Velocidades - {unit}")
            ax.set_xlabel("Velocidad (km/h)")
            ax.set_ylabel("Frecuencia")
            fig.tight_layout()
            fig.savefig(os.path.join(unit_dir, f"{unit}_speeds_histogram.png"), dpi=150)
            plt.close(fig)

    # --- Histograma de # estaciones por día ---
    # Asegurarse de que num_estaciones no cuenta NaN
    if not agg_est.empty:
        x = agg_est['num_estaciones'].astype(int)
        mn, mx = int(x.min()), int(x.max())
        if mn == mx:  # un solo valor
            bins = [mn - 0.5, mn + 0.5]
        else:
            bins = np.arange(mn - 0.5, mx + 1.5, 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        counts, edges, _ = ax.hist(x, bins=bins, alpha=0.8, edgecolor='black')
        ax.set_title("Histograma de Cantidad de Estaciones por Día")
        ax.set_xlabel("Cantidad de Estaciones"); ax.set_ylabel("Frecuencia")
        ax.set_xticks(range(mn, mx + 1))
        for c, left in zip(counts, edges[:-1]):
            if c > 0: ax.text(left + 0.5, c, f"{int(c)}", ha='center', va='bottom', fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(unit_dir, f"{unit}_stations_per_day_histogram.png"), dpi=150)
        plt.close(fig)

# -----------------------------
# Guardar mapa de días sin estaciones
# -----------------------------
def export_no_station_days_map(unit, out_dir, df, days_without_stations, max_points=200_000, zoom_start=13):
    """
    Genera un HTML con todos los puntos pertenecientes a días donde no se visitó ninguna estación.
    - df: dataframe ORIGINAL de la unidad (antes de filtrar días).
    - days_without_stations: serie/lista de fechas (date) donde num_estaciones == 0.
    - max_points: muestreo de seguridad para evitar HTML gigantes.
    """
    try:
        import folium
    except Exception as e:
        print(f"[{unit}] folium no disponible; no se generó el mapa de días sin estaciones. {e}")
        return None

    df_sin = df[df[COL_FECHA].dt.date.isin(days_without_stations)].dropna(subset=[COL_LAT, COL_LON])
    if df_sin.empty:
        print(f"[{unit}] Sin puntos en días sin estaciones; no se crea mapa.")
        return None

    # Muestreo para no generar HTML gigantes
    if len(df_sin) > max_points:
        df_sin = df_sin.sample(n=max_points, random_state=42)

    lat_mean = float(df_sin[COL_LAT].mean())
    lon_mean = float(df_sin[COL_LON].mean())
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=zoom_start, control_scale=True)

    # Puntos)
    for lat, lon in df_sin[[COL_LAT, COL_LON]].itertuples(index=False):
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=2,
            color="#d62728",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

    maps_dir = os.path.join(out_dir, "maps", f"{unit}")
    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(maps_dir, f"{unit}_sin_estaciones.html")
    m.save(out_path)
    print(f"[{unit}] Mapa de días sin estaciones: {out_path}")
    return out_path


# -----------------------------
# Proceso por unidad (end-to-end)
# -----------------------------
def process_unit(unit, input_dir, stations_df, tree, out_dir,
                 threshold_m=DEFAULT_THRESHOLD_M,
                 pause_s=DEFAULT_PAUSE_S,
                 pre_min=DEFAULT_PRE_MIN,
                 post_min=DEFAULT_POST_MIN,
                 use_idle_trim=DEFAULT_USE_IDLE_TRIM,
                 v0_kmh=DEFAULT_V0_KMH,
                 max_idle_s=DEFAULT_MAX_IDLE_S,
                 export_maps=False,
                 export_plots=False):
    in_path = os.path.join(input_dir, f"{unit}_joined.csv")
    if not os.path.exists(in_path):
        print(f"[{unit}] No se encontró {in_path}, se omite.")
        return None

    print(f"[{unit}] Leyendo {in_path}")
    df = load_unit_csv(in_path)

    # Asignación de estaciones
    df = assign_nearest_station(df, stations_df, tree, threshold_m=threshold_m)

    # Días sin estaciones
    df_filtrado, agg_est, dias_sin = filter_days_without_stations(df)
    
    # Histogramas por unidad
    if export_plots:
        export_histograms(unit, out_dir, df_original=df, agg_est=agg_est)
        
    # Mapa de días sin estaciones
    no_station_days_map_path = None
    if export_maps and len(dias_sin) > 0:
        no_station_days_map_path = export_no_station_days_map(unit, out_dir, df, dias_sin)

    # Segmentación a trips
    df_filtrado = segment_trips(df_filtrado, pause_s=pause_s)

    # Limpieza de trips
    df_limpio, post, drops, prev = clean_trips_by_stations(
        df_filtrado, pre_min, post_min, use_idle_trim, v0_kmh, max_idle_s
    )

    # Guardar outputs
    save_outputs(unit, out_dir, df_limpio, post, drops, prev)

    # Mapas
    if export_maps and not df_limpio.empty:
        export_trips_maps(unit, out_dir, df_limpio, no_station_days_map_path)

    # Devuelve resumen para consolidado global
    post['unit'] = unit
    drops['unit'] = unit
    prev['unit'] = unit
    return {'post': post, 'drops': drops, 'prev': prev}

# -----------------------------
# Descubrir unidades
# -----------------------------
def discover_units(input_dir):
    units = []
    for path in glob.glob(os.path.join(input_dir, "*_joined.csv")):
        base = os.path.basename(path)
        unit = base.replace("_joined.csv", "")
        units.append(unit)
    return sorted(units)

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Pipeline de limpieza de trayectorias Transmetro (todas las unidades).")
    ap.add_argument("--input-dir", required=True, help="Carpeta con *_joined.csv")
    ap.add_argument("--stations",   required=True, help="Ruta a Estaciones.xlsx")
    ap.add_argument("--out",        required=True, help="Carpeta de salida (se crearán subcarpetas por unidad)")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Procesar todas las unidades encontradas en input-dir")
    g.add_argument("--units", nargs="+", help="Lista de unidades a procesar (ej. u001 u049)")

    ap.add_argument("--threshold-m", type=int, default=DEFAULT_THRESHOLD_M, help="Umbral (m) para asignar estación")
    ap.add_argument("--pause-threshold", type=int, default=DEFAULT_PAUSE_S, help="Pausa (s) para cortar viajes")
    ap.add_argument("--pre-roll", type=int, default=DEFAULT_PRE_MIN, help="Minutos antes de 1ra estación a conservar")
    ap.add_argument("--post-roll", type=int, default=DEFAULT_POST_MIN, help="Minutos después de última estación a conservar")

    ap.add_argument("--idle-trim", action="store_true", help="Recortar colas estacionadas (requiere Velocidad)")
    ap.add_argument("--v0-kmh", type=float, default=DEFAULT_V0_KMH, help="Umbral de velocidad para considerar parado")
    ap.add_argument("--max-idle-s", type=int, default=DEFAULT_MAX_IDLE_S, help="Máximo estacionado a recortar (s)")

    ap.add_argument("--maps", action="store_true", help="Exportar mapas folium por trip")
    ap.add_argument("--plots", action="store_true", help="Guardar histogramas (velocidades y #estaciones por día)")
    ap.add_argument("--summary", action="store_true", help="Exportar CSVs consolidados (prev/post/drops) por todas las unidades")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    stations_df, tree = load_stations_and_tree(args.stations)

    if args.all:
        units = discover_units(args.input_dir)
    else:
        units = args.units

    all_prev = []; all_post = []; all_drops = []

    for unit in units:
        try:
            res = process_unit(
                unit=unit,
                input_dir=args.input_dir,
                stations_df=stations_df,
                tree=tree,
                out_dir=args.out,
                threshold_m=args.threshold_m,
                pause_s=args.pause_threshold,
                pre_min=args.pre_roll,
                post_min=args.post_roll,
                use_idle_trim=args.idle_trim,
                v0_kmh=args.v0_kmh,
                max_idle_s=args.max_idle_s,
                export_maps=args.maps,
                export_plots=args.plots
            )
            if res is not None:
                all_post.append(res['post'])
                all_prev.append(res['prev'])
                all_drops.append(res['drops'])
        except Exception as e:
            print(f"[{unit}] ERROR: {e}")

    if args.summary and (all_post or all_prev or all_drops):
        if all_prev:
            prev_glob = pd.concat(all_prev, ignore_index=True)
            prev_glob.to_csv(os.path.join(args.out, "GLOBAL_resumen_trips_pre.csv"), index=False)
        if all_post:
            post_glob = pd.concat(all_post, ignore_index=True)
            post_glob.to_csv(os.path.join(args.out, "GLOBAL_resumen_trips_post.csv"), index=False)
        if all_drops:
            drops_glob = pd.concat(all_drops, ignore_index=True)
            drops_glob.to_csv(os.path.join(args.out, "GLOBAL_drops_por_trip.csv"), index=False)
        print("[GLOBAL] Consolidados exportados.")

if __name__ == "__main__":
    main()
