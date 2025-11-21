'''Script de Ingeniería de características para segmentar viajes en líneas y direcciones usando el algoritmo de Viterbi,
para evaluar la adherencia de cada viaje a líneas candidatas mediante proyección sobre polilíneas densas definidas a partir de archivos CSV.'''

import os
import pandas as pd, numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt
import folium

# Funciones reutilizadas del script anterior (preparar rutas, proyección, etc.)
def meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    mlat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    mlon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return mlat, mlon

def ll_to_xy_m(lat, lon, lat0, lon0):
    mlat, mlon = meters_per_degree(lat0)
    return (lon - lon0)*mlon, (lat - lat0)*mlat

def xy_to_ll_m(x, y, lat0, lon0):
    mlat, mlon = meters_per_degree(lat0)
    return y/mlat + lat0, x/mlon + lon0

def cumulative_distances(x, y):
    dx = np.diff(x); dy = np.diff(y)
    return np.concatenate([[0.0], np.cumsum(np.sqrt(dx*dx + dy*dy))])

# Proyectar punto a la polilínea

def project_point_to_polyline(px, py, rx, ry, route_cum):
    """
    Devuelve (dist_m, s_m) proyectando (px,py) sobre todos los segmentos:
      - dist_m: distancia mínima (m)
      - s_m: abscisa (m) a lo largo de la polilínea
    """
    ax, ay = rx[:-1], ry[:-1]
    bx, by = rx[1:],  ry[1:]
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)
    seg2 = vx*vx + vy*vy
    # parámetro de proyección por segmento (clamp 0..1)
    t = np.where(seg2 > 0, (wx*vx + wy*vy)/seg2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    projx = ax + t*vx
    projy = ay + t*vy
    dx = px - projx
    dy = py - projy
    d  = np.sqrt(dx*dx + dy*dy)
    # s = acumulado al inicio del segmento + t * largo segmento
    seg_len = np.sqrt(seg2)
    s  = route_cum[:-1] + t*seg_len
    i  = int(np.argmin(d))
    return float(d[i]), float(s[i])

# Proyectar el trip a un candidato (línea, dirección)
def snap_trip_to_route(lat, lon, route):
    px, py = ll_to_xy_m(lat, lon, route["lat0"], route["lon0"])
    pairs = [project_point_to_polyline(px[i], py[i], route["rx"], route["ry"], route["route_cum"])
             for i in range(len(px))]
    dist_m = np.fromiter((p[0] for p in pairs), dtype=float, count=len(pairs))
    s_m    = np.fromiter((p[1] for p in pairs), dtype=float, count=len(pairs))
    return dist_m, s_m

def load_line_polyline_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Normaliza nombre de columnas
    needed = {"name","dir","lat","lon","station"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Faltan columnas {needed - set(df.columns)} en {csv_path}")

    # Vértice es estación cuando "station" no es nula
    if "station" in df.columns:
        df["kind"] = np.where(df["station"].notna(), "station", "vertex")
    else:
        raise ValueError("Incluir 'station' para diferenciar estaciones.")

    if "kind" not in df.columns:
        df["kind"] = np.where(df["is_station"].astype(bool), "station", "vertex")

    # Utilizar el orden original del CSV
    df["order"] = np.arange(len(df), dtype=int)

    geoms = {}

    for (linea, d), g in df.sort_values("order").groupby(["name","dir"], sort=False):
        g = g.copy()

        latv = g["lat"].astype(float).to_numpy()
        lonv = g["lon"].astype(float).to_numpy()
        lat0, lon0 = float(latv.mean()), float(lonv.mean())

        rx, ry = ll_to_xy_m(latv, lonv, lat0, lon0)
        route_cum = cumulative_distances(rx, ry)
        
        # Ruta circular?
        is_circular = g["dir"].iloc[0] == 'CIRCULAR'

        route = dict(
            rx=rx, ry=ry, route_cum=route_cum,
            lat0=lat0, lon0=lon0,
            length_m=float(route_cum[-1]),
            is_circular=is_circular
        )
        
        geoms[(linea, d)] = route
        
    return geoms

# Intento de uso con viterbi:

import numpy as np
import pandas as pd

def build_emission_costs(lat, lon, states, geoms, Dmax=120.0):
    """
    states: lista de claves (line, dir)
    return: costs (N,K), mantener per-state s_m/dist_m
    """
    N = len(lat); K = len(states)
    costs = np.full((N, K), 1e9, float)
    s_by_state = [None]*K
    d_by_state = [None]*K
    for k, key in enumerate(states):
        route = geoms.get(key)
        if route is None: continue
        dist_m, s_m = snap_trip_to_route(lat, lon, route)
        # emisión robusta: capear distancia
        c = np.minimum(dist_m, Dmax)
        costs[:, k] = c*c
        s_by_state[k] = s_m
        d_by_state[k] = dist_m
    return costs, s_by_state, d_by_state

def near_terminal_mask(route, s_arr, window_m=50.0):
    if route is None or s_arr is None: 
        return np.zeros_like(s_arr, dtype=bool)
    return (route["length_m"] - s_arr) <= window_m  # cerca del final

def viterbi_path(costs, switch_penalty=60.0, dyn_penalty=None):
    """
    costs: (N,K)
    dyn_penalty: callable(i, j, s) -> penalización extra por cambiar de j->s en i (puede ser 0)
    """
    N, K = costs.shape
    dp = np.empty((N, K), float); dp[0] = costs[0]
    prev = np.full((N, K), -1, int)
    for i in range(1, N):
        # trans base: penaliza cambiar; 0 si se queda
        K = costs.shape[1]
        # matriz KxK: costo de venir de cada estado j a cada estado s
        base = np.repeat(dp[i-1][:, None], K, axis=1) + switch_penalty  # (K,K)
        # quedarse en el mismo estado NO paga penalización de cambio
        base[np.arange(K), np.arange(K)] = dp[i-1]                      # diag = dp[i-1]

        if dyn_penalty is not None:
            base = base + dyn_penalty(i)  # suma (K,K) -> (K,K)
        j_best = np.argmin(base, axis=0)     # mejor origen para cada destino s
        dp[i] = costs[i] + base[j_best, np.arange(K)]
        prev[i] = j_best
    path = np.empty(N, int)
    path[-1] = int(np.argmin(dp[-1]))
    for i in range(N-1, 0, -1):
        path[i-1] = prev[i, path[i]]
    return path

def sequence_to_segments(path, min_pts=8, min_progress=300.0, s_by_state=None, states=None):
    """
    Colapsa el path en segmentos y filtra cortos o con poco progreso.
    """
    N = len(path)
    segs = []
    i0 = 0
    for i in range(1, N+1):
        if i==N or path[i] != path[i-1]:
            k = path[i-1]
            i1 = i-1
            keep = True
            if (i1 - i0 + 1) < min_pts: 
                keep = False
            if keep and s_by_state is not None:
                s = s_by_state[k]
                if s is not None:
                    prog = float(np.nanmax(s[i0:i1+1]) - np.nanmin(s[i0:i1+1]))
                    if prog < min_progress:
                        keep = False
            if keep:
                line, d = states[k] if states else (None, None)
                segs.append({"state_k": int(k), "LINEA": line, "DIR": d, "i0": int(i0), "i1": int(i1)})
            i0 = i
    return pd.DataFrame(segs)

def segment_trip_with_viterbi(trip_df, geoms, shortlist_states, params=None):
    """
    trip_df: DataFrame de 1 trip ordenado por Fecha (tiene Latitud/Longitud)
    shortlist_states: lista de (line, dir) permitidos para este día/trip
    params: dict opcional
    """
    P = dict(Dmax=120.0, switch_penalty=60.0, term_window=50.0,
             min_pts=8, min_progress=300.0)
    if params: P.update(params)

    lat = trip_df["Latitud"].to_numpy(float)
    lon = trip_df["Longitud"].to_numpy(float)

    costs, s_by_state, _ = build_emission_costs(lat, lon, shortlist_states, geoms, Dmax=P["Dmax"])

    # penalización dinámica: bajar costo de cambio solo cerca de terminales
    routes = [geoms.get(st) for st in shortlist_states]
    near_end_flags = [near_terminal_mask(rt, s_by_state[i], P["term_window"]) if s_by_state[i] is not None else None
                      for i, rt in enumerate(routes)]

    def dyn_penalty(i):
        # matriz KxK con extras por cambiar de j->s en i
        K = len(shortlist_states)
        extra = np.zeros((K, K), float)
        for j in range(K):
            for s in range(K):
                if j == s: 
                    continue
                # por defecto: nada
                scale = 1.0
                # si estoy cerca del final del estado j, bajar penalización
                flag = (near_end_flags[j][i] if (near_end_flags[j] is not None and i < len(near_end_flags[j])) else False)
                if flag:
                    scale = 0.3  # permite el cambio
                # si cambio entre IDA<->VUELTA de la MISMA línea: solo barato cerca de terminal
                lj, dj = shortlist_states[j]
                ls, ds = shortlist_states[s]
                if lj == ls and (dj, ds) in {("IDA","VUELTA"), ("VUELTA","IDA")}:
                    # si NO estoy en terminal, subir un poco la penalización
                    if not flag:
                        scale = 1.2
                extra[j, s] = (scale - 1.0) * P["switch_penalty"]
        return extra

    path = viterbi_path(costs, switch_penalty=P["switch_penalty"], dyn_penalty=dyn_penalty)
    segs = sequence_to_segments(path, min_pts=P["min_pts"], min_progress=P["min_progress"],
                                s_by_state=s_by_state, states=shortlist_states)
    return segs, path


# Visualizar resultados
# Graficar la polilínea del inicio de cada viaje con folium

def plot_trip_with_adherence(g_trip, met, outfile="trip_map.html"):
    """
    g_trip: DataFrame solo del trip (ya filtrado), ordenado por Fecha
    met: dict con t_start_ts, t_end_ts, idx_start, idx_end
    """
    import folium

    # Localiza la posición (iloc) dentro de g_trip a partir de los índices reales
    pos0 = int(met["t_start"])
    pos1 = int(met["t_end"])

    # Rebanadas seguras
    g_pre  = g_trip.iloc[:pos0]
    g_run  = g_trip.iloc[pos0:pos1+1]
    g_post = g_trip.iloc[pos1+1:]

    # (Opcional) sanity checks útiles
    print("Check tiempos:",
          g_run["Fecha"].iloc[0], "vs", met["t_start_ts"],
          "|", g_run["Fecha"].iloc[-1], "vs", met["t_end_ts"])

    # Centro del mapa
    center = [g_run["Latitud"].iloc[0], g_run["Longitud"].iloc[0]]
    m = folium.Map(location=center, zoom_start=13)

    def add_poly(df, color, weight=3, opacity=0.9, name="segment"):
        if df.empty: return
        coords = df[["Latitud","Longitud"]].values.tolist()
        folium.PolyLine(coords, color=color, weight=weight, opacity=opacity, tooltip=name).add_to(m)

    # Pinta “antes”, “racha adherida”, “después”
    add_poly(g_pre,  "#e31717", 2, 0.6, "antes")
    add_poly(g_run,  "#23d30f", 4, 0.9, "adherido (t_start → t_end)")
    add_poly(g_post, "#650fd6", 2, 0.6, "después")

    # Marcas de inicio/fin de la racha
    folium.Marker(
        [g_run["Latitud"].iloc[0], g_run["Longitud"].iloc[0]],
        popup=f"t_start ({met['t_start_ts']})",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    folium.Marker(
        [g_run["Latitud"].iloc[-1], g_run["Longitud"].iloc[-1]],
        popup=f"t_end ({met['t_end_ts']})",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    m.save(outfile)
    print("Mapa guardado en:", outfile)

# --- Ejecución principal ---

DIST_THRESH_M   = 50.0  # distancia máxima a la ruta para considerar "pegado"
MIN_POINTS      = 8       # puntos mínimos en la racha inicial
MIN_PROGRESS_M  = 300.0   # avance mínimo (m) a lo largo de la ruta en la racha
MAX_BACKTRACK_M = 80.0    # retroceso máximo permitido entre puntos consecutivos (m)
FRAC_WITHIN     = 0.9    # % de puntos dentro del umbral en la ventana

def add_line_feature(unit):
    
    print(f'--- Procesando unidad {unit} ---')
    
    # Constantes de unidad y líneas
    UNIT = unit
    TRIPS_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\clean_data\\{UNIT}\\{UNIT}_clean_trips.csv"
    LINES = ['Linea_1', 'Linea_2', 'Linea_6', 'Linea_7', 'Linea_12', 'Linea_13-A', 'Linea_13-B', 'Linea_18-A', 'Linea_18-B']
    
    # Cargar trips
    trips_df = pd.read_csv(TRIPS_CSV)
    trips_df["Fecha"] = pd.to_datetime(trips_df["Fecha"], errors="coerce")
    trips_df = trips_df.sort_values(["trip_id","Fecha"])

    if trips_df.empty:
        print(f'No hay datos para la unidad {UNIT}. Se omite el procesamiento.')
        return
    
    # Trip específico para pruebas
    """ df = trips_df[trips_df["trip_id"] == 3]
    if df.empty:
        print(f"El trip_id '1' no existe en {TRIPS_CSV}. Fin.")
        return """
    
    # Cargar geometrías de todas las líneas
    geoms = {}
    for line in LINES:
        path = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\complete_lines\\{line}_with_direction.csv"
        geoms.update(load_line_polyline_csv(path))
        
    # Analizar cada trip por separado
    rows = []
    for tid, g in trips_df.groupby("trip_id", sort=False):
        g = g.sort_values("Fecha")
        unique_stations_visited = g["estacion_cercana"].dropna().astype(str).unique().tolist()
        lat = g["Latitud"].to_numpy(dtype=float)
        lon = g["Longitud"].to_numpy(dtype=float)
        ts  = g["Fecha"]                      # Series alineada con lat/lon
        idx = g.index                         # índices reales en df
        
        # Preselección de rutas a probar:
        cand_list = []

        if 'SAN RAFAEL' in unique_stations_visited or 'PARAÍSO' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_18-b'.casefold() ]
            
        if 'ATLÁNTIDA' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_18-a'.casefold() ]

        if 'CEJUSA ANDÉN SUR ' in unique_stations_visited or 'CEJUSA ANDÉN NORTE' in unique_stations_visited or 'VILLA LINDA ANDÉN NORTE' in unique_stations_visited or 'VILLA LINDA ANDÉN SUR' in unique_stations_visited or 'ROOSEVELT ANDÉN NORTE' in unique_stations_visited or 'ROOSEVELT ANDÉN SUR' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_7'.casefold()]

        if 'JOCOTENANGO' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_2'.casefold()]

        if 'CENTRO ZONA 6' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_6'.casefold()]
            
        if 'SAN AGUSTIN' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_1'.casefold()]
            
        if 'PLAZA BERLÍN' in unique_stations_visited or 'JUAN PABLO II' in unique_stations_visited or 'PLAZA ARGENTINA' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_13-b'.casefold()]

        if 'EXPOSICIÓN' in unique_stations_visited or 'TERMINAL' in unique_stations_visited or 'INDUSTRIA' in unique_stations_visited or 'TIVOLI' in unique_stations_visited or 'TORRE DEL REFORMADOR' in unique_stations_visited or 'SEIS 26' in unique_stations_visited:
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_13-a'.casefold()]
            
        # Fallback: estaciones de Línea 12 (estaciones compartidas por otras líneas)
        if not cand_list and ('DON BOSCO' in unique_stations_visited or 'TRÉBOL' in unique_stations_visited or 'EL CARMEN' in unique_stations_visited):
            cand_list += [ key for key in geoms.keys() if key[0].strip().casefold() == 'linea_12'.casefold()]
            
        # si no hay shortlist, prueba TODAS las rutas (menos preciso)
        if not cand_list:
            cand_list = [ key for key in geoms.keys() ]
            
        shortlist_states = list(dict.fromkeys(cand_list))  # quita duplicados, conserva orden

        # --- corre Viterbi sobre todo el trip ---
        params = dict(Dmax=120.0, switch_penalty=60.0, term_window=50.0,
                    min_pts=8, min_progress=300.0)
        segs, path = segment_trip_with_viterbi(g, geoms, shortlist_states, params=params)

        """ if segs.empty:
            # si no se formó ningún segmento “estable”, deja fila nula (como antes)
            rows.append({"trip_id": tid, "LINEA": None, "DIR": None,
                        "t_start": None, "t_start_ts": None, "idx_start": None,
                        "t_end": None, "t_end_ts": None, "idx_end": None,
                        "progress_m": 0.0, "mean_dev_m": np.nan,
                        "frac_in": 0.0, "dur_pts": 0, "score": -1e9})
            continue """
        
        # 1) Mapas de estado -> LINEA/DIR, y “paths” por muestra
        state2line = np.array([st[0] for st in shortlist_states], dtype=object)
        state2dir  = np.array([st[1] for st in shortlist_states], dtype=object)

        line_path = state2line[path]   # LÍNEA por punto
        dir_path  = state2dir[path]    # DIR por punto (solo para elegir DIR inicial del bloque)

        # 2) Colapsar runs por LÍNEA (ignorando DIR)
        def collapse_runs(vals):
            segs_line = []
            i0 = 0
            for i in range(1, len(vals)+1):
                if i == len(vals) or vals[i] != vals[i-1]:
                    segs_line.append((vals[i-1], i0, i-1))
                    i0 = i
            return segs_line

        runs_line = collapse_runs(line_path)

        # 3) Histéresis mínima por LÍNEA para filtrar chispazos
        min_pts_line = 50
        runs_line = [(ln,i0,i1) for (ln,i0,i1) in runs_line if (i1 - i0 + 1) >= min_pts_line]

        # 4) DIR inicial del bloque: mayoría en una ventana inicial
        W = 40  # ajusta
        def pick_dir_init(i0, i1):
            j1 = min(i1, i0 + W - 1)
            sub = dir_path[i0:j1+1]
            if len(sub) == 0:
                return None
            vals, cnts = np.unique(sub, return_counts=True)
            return str(vals[np.argmax(cnts)])

        # 5) Generar filas “solo cambios de LÍNEA” (con DIR_init)
        previous_line = None
        for (ln, i0, i1) in runs_line:
            if ln == previous_line:
                continue
            previous_line = ln
            rows.append({
                "trip_id": tid,
                "LINEA": ln,
                "DIR_init": pick_dir_init(i0, i1),  # solo para arrancar tu módulo de siguiente estación
                "t_start": float(i0),
                "t_end": float(i1),
                "t_start_ts": ts.iloc[i0],
                "t_end_ts": ts.iloc[i1],
                "idx_start": int(idx[i0]),
                "idx_end": int(idx[i1]),
                "dur_pts": int(i1 - i0 + 1)
            })
            
        # --- convierte segmentos (i0,i1) a índices y métricas ---
        # prepara emisiones por estado para medir mean_dev/frac_in/progreso
        """ costs, s_by_state, d_by_state = build_emission_costs(lat, lon, shortlist_states, geoms, Dmax=params["Dmax"])

        for _, seg in segs.iterrows():
            k = int(seg["state_k"])
            line_k, dir_k = shortlist_states[k]
            i0, i1 = int(seg["i0"]), int(seg["i1"])
            # mapea a índice/timestamp original
            idx_start = int(idx[i0]); idx_end = int(idx[i1])
            t_start_ts = ts.iloc[i0]; t_end_ts = ts.iloc[i1]

            s_seg = s_by_state[k][i0:i1+1]
            d_seg = d_by_state[k][i0:i1+1]
            progress = float(np.nanmax(s_seg) - np.nanmin(s_seg)) if s_seg.size else 0.0
            mean_dev = float(np.nanmean(d_seg)) if d_seg.size else np.nan
            # fracción dentro del umbral de “pegado”
            inside_mask = np.isfinite(d_seg)
            frac_in = float(np.mean(d_seg[inside_mask] <= DIST_THRESH_M)) if np.any(inside_mask) else 0.0

            rows.append({
                "trip_id": tid, "LINEA": line_k, "DIR": dir_k,
                "t_start": i0, "t_end": i1,
                "t_start_ts": t_start_ts, "t_end_ts": t_end_ts,
                "idx_start": idx_start, "idx_end": idx_end,
                "progress_m": progress, "mean_dev_m": mean_dev,
                "frac_in": frac_in, "dur_pts": int(i1 - i0 + 1),
                # un score sencillo por si quieres priorizar
                "score": float(2.5*progress + 1.5*frac_in - mean_dev/DIST_THRESH_M)
            }) """
    
        # Debug: guardar path_df
        """ state_names = [f"{l}|{d}" for (l,d) in shortlist_states]
        path_df = pd.DataFrame({
            "trip_id": tid,
            "index": idx.values,
            "Fecha": ts.values,
            "state_k": path,
            "state_name": [state_names[k] for k in path]
        }) """
        
        """ OUT_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\tests\\{UNIT}\\{UNIT}_trip_{tid}_viterbi_path.csv"
        
        # Crear carpeta si no existe
        import os
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

        path_df.to_csv(OUT_CSV, index=False) """

    # Guardar resultados
    trips_with_viterbi = pd.DataFrame(rows)
    
    # Ordenar por trip_id y t_start
    trips_with_viterbi = trips_with_viterbi.sort_values(["trip_id","t_start"])

    OUT_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{UNIT}\\{UNIT}_trips_with_viterbi.csv"

    # Crear carpeta si no existe
    import os
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    trips_with_viterbi.to_csv(OUT_CSV, index=False)
    
    
    print(f"Resultados guardados en: {OUT_CSV}")
    
    # Visualizar resultados (opcional)
    """ for _, row in trips_with_viterbi.iterrows():
        if pd.isna(row["LINEA"]):
            print(f"Trip {row['trip_id']}: No se encontró línea candidata.")
            continue
        tid = row["trip_id"]
        g_trip = df[df["trip_id"] == tid].sort_values("Fecha")
        if g_trip.empty:
            print(f"Trip {tid} no encontrado en los datos originales.")
            continue
        met = {
            "t_start": row["t_start"],
            "t_end": row["t_end"],
            "t_start_ts": row["t_start_ts"],
            "t_end_ts": row["t_end_ts"],
            "idx_start": row["idx_start"],
            "idx_end": row["idx_end"]
        }
        outfile = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\tests\\{UNIT}\\trip_{tid}_line_{row['LINEA']}_dir_{row['DIR']}.html"
        plot_trip_with_adherence(g_trip, met, outfile=outfile) """

# --- Ejecución principal ---

if __name__ == "__main__":
    print('=== Iniciando procesamiento de líneas con Viterbi ===')
    # Encontrar todas las unidades (carpetas en clean_data)
    CLEAN_DATA_DIR = Path("D:/2025/UVG/Tesis/repos/backend/clean_data")
    if not CLEAN_DATA_DIR.exists():
        print(f"No existe el directorio {CLEAN_DATA_DIR}. Fin.")
    units = [p.name for p in CLEAN_DATA_DIR.iterdir() if p.is_dir() and p.name != "maps"]
    if not units:
        print(f"No se encontraron unidades en {CLEAN_DATA_DIR}. Fin.")
    for unit in units:
        # Si ya existe el archivo de salida, omitir
        out_csv = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{unit}\\{unit}_trips_with_viterbi.csv"
        if os.path.exists(out_csv):
            print(f"El archivo {out_csv} ya existe. Se omite la unidad {unit}.")
            continue
        add_line_feature(unit)
    for unit in units:
        add_line_feature(unit)