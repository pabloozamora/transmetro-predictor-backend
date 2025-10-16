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

# Encontrar el primer punto adherido a la línea
def find_first_adherent_index(dist_m, s_m,
                              distance_thresh_m=200.0,
                              min_points=8,
                              min_progress_m=600.0,
                              max_backtrack_m=80.0,
                              window_points=None,
                              frac_within=0.8,
                              route_length=None,
                              is_circular=False):
    
    if window_points is None: window_points = min_points
    n = len(dist_m)
    
    def diffs_forward(s):
        ds = np.diff(s)
        if is_circular and (route_length is not None) and (route_length > 0):
            wrap_mask = ds < -0.5*route_length
            ds = np.where(wrap_mask, ds + route_length, ds)
        return ds
    
    def good(i0, i1):
        d = dist_m[i0:i1]; s = s_m[i0:i1]
        inside = np.isfinite(d) & (d <= distance_thresh_m)
        
        if inside.size == 0 or np.mean(inside) < frac_within: return False
        
        s_ok = np.isfinite(s)
        if not np.any(s_ok): return False
        
        s2 = s[s_ok]
        ds = diffs_forward(s2)
        back = np.maximum(0.0, -ds)
        
        if np.any(back > max_backtrack_m): return False
        prog = float(np.nanmax(s2) - np.nanmin(s2))
        return prog >= min_progress_m
    
    for start in range(0, n - window_points + 1):
        if good(start, start + window_points):
            j = start + window_points
            while j < n and good(start, j): j += 1
            return start, j - 1
    return None, None

# Proyectar el trip a un candidato (línea, dirección)
def snap_trip_to_route(lat, lon, route):
    px, py = ll_to_xy_m(lat, lon, route["lat0"], route["lon0"])
    pairs = [project_point_to_polyline(px[i], py[i], route["rx"], route["ry"], route["route_cum"])
             for i in range(len(px))]
    dist_m = np.fromiter((p[0] for p in pairs), dtype=float, count=len(pairs))
    s_m    = np.fromiter((p[1] for p in pairs), dtype=float, count=len(pairs))
    return dist_m, s_m

# Métricas de adhesión por cada candidato del trip
def adhesion_metrics_for_candidate(lat, lon, route,
                                   DIST_THRESH_M=200.0,
                                   MIN_POINTS=6,
                                   MIN_PROGRESS_M=200.0,
                                   MAX_BACKTRACK_M=80.0,
                                   FRAC_WITHIN=0.8,
                                   ts=None,
                                   idx=None):
    dist_m, s_m = snap_trip_to_route(lat, lon, route)
    t0, t1 = find_first_adherent_index(dist_m, s_m,
                                       distance_thresh_m=DIST_THRESH_M,
                                       min_points=MIN_POINTS,
                                       min_progress_m=MIN_PROGRESS_M,
                                       max_backtrack_m=MAX_BACKTRACK_M,
                                       frac_within=FRAC_WITHIN,
                                       route_length=route["length_m"],
                                       is_circular=route["is_circular"])
    if t0 is None:
        return None  # no hay adhesión sostenida
    
    run = slice(t0, t1+1)
    mean_dev = float(np.nanmean(dist_m[run]))
    frac_in  = float(np.mean(dist_m[run] <= DIST_THRESH_M))
    progress = float(np.nanmax(s_m[run]) - np.nanmin(s_m[run]))
    dur_pts  = int(t1 - t0 + 1)
    
    result = dict(t_start=t0, t_end=t1, mean_dev=mean_dev,
                frac_in=frac_in, progress=progress, dur_pts=dur_pts,
                route_len=route["length_m"])
    
    # timestamps e índices originales
    if ts is not None:
        ts = pd.to_datetime(ts, errors="coerce")
        result["t_start_ts"] = ts.iloc[t0]
        result["t_end_ts"]   = ts.iloc[t1]
    if idx is not None:
        idx = np.asarray(idx)
        result["idx_start"] = int(idx[t0])
        result["idx_end"]   = int(idx[t1])

    return result

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
    """ df = trips_df[trips_df["trip_id"] == 6]
    if df.empty:
        print(f"El trip_id '6' no existe en {TRIPS_CSV}. Fin.")
        return """
    
    # Cargar geometrías de todas las líneas
    geoms = {}
    for line in LINES:
        path = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\complete_lines\\{line}_with_direction.csv"
        geoms.update(load_line_polyline_csv(path))
        
    rows = []
    for tid, g in trips_df.groupby("trip_id", sort=False):
        g = g.sort_values("Fecha")
        unique_stations_visited = g["estacion_cercana"].dropna().astype(str).unique().tolist()
        lat = g["Latitud"].to_numpy(dtype=float)
        lon = g["Longitud"].to_numpy(dtype=float)
        ts  = g["Fecha"]                      # Series alineada con lat/lon
        idx = g.index                         # índices reales en df
        
        # Preselección de rutas a probar:
        cand_list = None

        if 'SAN RAFAEL' in unique_stations_visited or 'PARAÍSO' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_18-B'.casefold()]
            if 'CEJUSA ANDÉN SUR' in unique_stations_visited or 'JOCOTENANGO' in unique_stations_visited or 'CENTRO ZONA 6' in unique_stations_visited or 'EXPOSICIÓN' in unique_stations_visited:
                continue # si hay estaciones de otras líneas, omite este trip (multi-línea probable)
            
        elif 'ATLÁNTIDA' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_18-A'.casefold()]
            if 'CEJUSA ANDÉN SUR' in unique_stations_visited or 'JOCOTENANGO' in unique_stations_visited or 'CENTRO ZONA 6' in unique_stations_visited or 'EXPOSICIÓN' in unique_stations_visited:
                continue # si hay estaciones de otras líneas, omite este trip (multi-línea probable)

        elif 'CEJUSA ANDÉN SUR' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_7'.casefold()]
            if 'JOCOTENANGO' in unique_stations_visited or 'CENTRO ZONA 6' in unique_stations_visited or 'EXPOSICIÓN' in unique_stations_visited:
                continue # si hay estaciones de otras líneas, omite este trip (multi-línea probable)

        elif 'JOCOTENANGO' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_2'.casefold()]
            if 'CENTRO ZONA 6' in unique_stations_visited or 'EXPOSICIÓN' in unique_stations_visited:
                continue # si hay estaciones de otras líneas, omite este trip (multi-línea probable)

        elif 'CENTRO ZONA 6' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_6'.casefold()]
            if 'EXPOSICIÓN' in unique_stations_visited:
                continue # si hay estaciones de otras líneas, omite este trip (multi-línea probable)

        elif 'EXPOSICIÓN' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_13-A'.casefold()] + [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea_13-B'.casefold()]

        # si no hay shortlist, prueba TODAS las rutas (puede ser más lento)
        if not cand_list:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() ]
            
        scored = []
        for (linea, d), _, _, _ in cand_list:
            route = geoms.get((linea, d))
            if route is None: 
                continue
            
            min_points = MIN_POINTS
            min_progress = MIN_PROGRESS_M
            max_backtrack = MAX_BACKTRACK_M
            
            # AJUSTE PARA CASOS ESPECIALES (mucho ruido)
            # rutas muy cortas: relaja mínimo de puntos y distancia mínima
            if len(route["rx"]) < MIN_POINTS:
                min_points = len(route["rx"])
                min_progress = 300
                max_backtrack = 150
                
            
            met = adhesion_metrics_for_candidate(lat, lon, route,
                                                DIST_THRESH_M, min_points, min_progress,
                                                max_backtrack, FRAC_WITHIN, ts=ts, idx=idx)
            if met is None:
                continue

            # Score adhesión:
            #   - mucha duración y progreso (normalizado por largo ruta)
            #   - alta fracción dentro del umbral
            #   - poca desviación promedio
            #   - inicio temprano favorecido
            dur = met["dur_pts"]
            prog_norm = met["progress"]
            score = (2.5*prog_norm) + (1.5*met["frac_in"]) + (0.5*dur/len(lat)) - (met["mean_dev"]/DIST_THRESH_M) - (0.3*met["t_start"]/max(len(lat),1))

            scored.append({
                "trip_id": tid, "LINEA": linea, "DIR": d,
                "t_start": met["t_start"], "t_end": met["t_end"],
                "t_start_ts": met.get("t_start_ts"), "t_end_ts": met.get("t_end_ts"),
                "idx_start": met.get("idx_start"), "idx_end": met.get("idx_end"),
                "progress_m": met["progress"], "mean_dev_m": met["mean_dev"],
                "frac_in": met["frac_in"], "dur_pts": dur, "score": float(score)
            })

        if scored:
            scored.sort(key=lambda r: (r["LINEA"], r["DIR"]))
            best_by_line = {}
            for r in scored:
                key = r["LINEA"]
                if key not in best_by_line:
                    best_by_line[key] = r
                else:
                    # si es la misma línea, favorece el inicio más temprano
                    cur = best_by_line[key]
                    if (r["t_start"] < cur["t_start"]):
                        best_by_line[key] = r
                        
            best = max(best_by_line.values(), key=lambda r: r["score"])
            rows.append(best)
        else:
            rows.append({"trip_id": tid, "LINEA": None, "DIR": None,
                        "t_start": None, "t_start_ts": None, "idx_start": None,
                        "t_end": None, "t_end_ts": None, "idx_end": None,
                        "progress_m": 0.0, "mean_dev_m": np.nan,
                        "frac_in": 0.0, "dur_pts": 0, "score": -1e9})

    route_scores_df = pd.DataFrame(rows)
    
    # Ordenar resultados por trip_id numéricamente
    route_scores_df["trip_id_num"] = pd.to_numeric(route_scores_df["trip_id"], errors="coerce")
    route_scores_df = route_scores_df.sort_values("trip_id_num").drop(columns=["trip_id_num"])

    # Guardar resultados
    OUT_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\tests\\{UNIT}\\{UNIT}_trip_routes_test.csv"

    # Crear carpeta si no existe
    import os
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    route_scores_df.to_csv(OUT_CSV, index=False)
    
    # Guardar trips con t_start > 300 (muy probables de ser multi-línea)
    multiline_trips = route_scores_df[route_scores_df["t_start"] > 300]
    OUT_MULTI_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\tests\\{UNIT}\\{UNIT}_multiline_trips_test.csv"
    multiline_trips.to_csv(OUT_MULTI_CSV, index=False)
    
    print(f"Resultados guardados en: {OUT_CSV} y {OUT_MULTI_CSV}")
    
    # Visualizar resultados
    """ for _, row in route_scores_df.iterrows():
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

# Ejecución de prueba
UNIT = 'u204'
add_line_feature(UNIT)