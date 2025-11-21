# -- FEATURE ENGINEERING ---
import pandas as pd, numpy as np, math, os, glob

# Feature 1: adhesión a ruta

DIST_THRESH_M   = 200.0   # distancia máxima a la ruta para considerar "pegado"
MIN_POINTS      = 8       # puntos mínimos en la racha inicial
MIN_PROGRESS_M  = 600.0   # avance mínimo (m) a lo largo de la ruta en la racha
MAX_BACKTRACK_M = 80.0    # retroceso máximo permitido entre puntos consecutivos (m)
FRAC_WITHIN     = 0.8     # % de puntos dentro del umbral en la ventana

def prepare_route_geoms(df_stations):
    geoms = {}  # (LINEA, DIR) -> dict(rx, ry, route_cum, length_m, lat0, lon0, is_circular)
    for (linea, d), g in df_stations.sort_values("ORDEN").groupby(["LINEA","DIR"], sort=False):
        
        # usa columnas lat/lon
        latv = g[[c for c in g.columns if c.upper()=="LAT"][0]].astype(float).to_numpy()
        lonv = g[[c for c in g.columns if c.upper()=="LON"][0]].astype(float).to_numpy()

        if len(latv) < 2: 
            continue

        lat0, lon0 = float(latv.mean()), float(lonv.mean())
        rx, ry = ll_to_xy_m(latv, lonv, lat0, lon0)         # ya tienes ll_to_xy_m
        route_cum = cumulative_distances(rx, ry)            # ya tienes cumulative_distances
        length_m = float(route_cum[-1])
        
        # ruta cirucular?
        is_circ = str(d).upper()=="CIRCULAR"

        geoms[(linea, d)] = dict(rx=rx, ry=ry, route_cum=route_cum, length_m=length_m,
                                 lat0=lat0, lon0=lon0, is_circular=is_circ)
        
    return geoms

def meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    mlat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    mlon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return mlat, mlon

def ll_to_xy_m(lat, lon, lat0, lon0):
    mlat, mlon = meters_per_degree(lat0)
    return (lon - lon0)*mlon, (lat - lat0)*mlat

def cumulative_distances(x, y):
    dx = np.diff(x); dy = np.diff(y)
    return np.concatenate([[0.0], np.cumsum(np.sqrt(dx*dx + dy*dy))])

def project_point_to_polyline(px, py, rx, ry, route_cum):
    if len(rx) < 2: return float('inf'), float('nan')
    dx = np.diff(rx); dy = np.diff(ry)
    seg2 = dx*dx + dy*dy
    ax = rx[:-1]; ay = ry[:-1]
    pax = px - ax;  pay = py - ay
    t = np.divide(pax*dx + pay*dy, seg2, out=np.zeros_like(seg2), where=seg2>0)
    t = np.clip(t, 0, 1)
    projx = ax + t*dx; projy = ay + t*dy
    d2 = (px - projx)**2 + (py - projy)**2
    i = int(np.argmin(d2))
    s = float(route_cum[i] + t[i]*(np.sqrt(seg2[i]) if seg2[i]>0 else 0))
    return float(np.sqrt(d2[i])), s

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

def adhesion_metrics_for_candidate(lat, lon, route,
                                   DIST_THRESH_M=200.0,
                                   MIN_POINTS=8,
                                   MIN_PROGRESS_M=600.0,
                                   MAX_BACKTRACK_M=80.0,
                                   FRAC_WITHIN=0.8,
                                   ts=None,
                                   idx=None):
    dist_m, s_m = snap_track_to_route(lat, lon, route)
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

# --- proyectar el trip a una ruta
def snap_track_to_route(lat, lon, route):
    px, py = ll_to_xy_m(lat, lon, route["lat0"], route["lon0"])
    pairs = [project_point_to_polyline(px[i], py[i], route["rx"], route["ry"], route["route_cum"])
             for i in range(len(px))]
    dist_m = np.fromiter((p[0] for p in pairs), dtype=float, count=len(pairs))
    s_m    = np.fromiter((p[1] for p in pairs), dtype=float, count=len(pairs))
    return dist_m, s_m

# Ejecución de feature 1 para una unidad

def add_adhesion_feature(unit, stations_ord):
        
    # Parámetros y rutas
    UNIT = unit
    TRACK_CSV   = f"../clean_data/{UNIT}/{UNIT}_clean_trips.csv"

    # Carga trips
    df = pd.read_csv(TRACK_CSV, dtype={"trip_id": str})
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values(["trip_id","Fecha"])
    
    # Si el dataframe está vacío, salir
    if df.empty:
        print(f'No hay datos para la unidad {UNIT}. Se omite el procesamiento.')
        return

    # Trip específico para pruebas
    # df = df[df["trip_id"] == "407"]

    # Secuencia por trip de estaciones cercanas
    trip_seqs = (
        df.dropna(subset=["estacion_cercana"])
        .assign(estacion_cercana=lambda x: x["estacion_cercana"].astype(str))
        .groupby("trip_id")["estacion_cercana"]
        .apply(list)
        .to_dict()
    )

    # Por trip, hacer un shortlist de K rutas candidatas:
    """ K = 3
    trip_candidates = {}  # tid -> list[(key=(LINEA,DIR), length, coverage, last_prog)]
    for tid, est_seq in trip_seqs.items():
        cands = []
        for key, seq in route_map.items():
            positions = [station_pos.get((key[0], key[1], s)) for s in est_seq]
            mapped = [p for p in positions if p is not None]
            if not mapped:
                continue
            length, lis_idx = lis_indices(mapped)
            coverage = len(set([positions[i] for i in range(len(positions)) if positions[i] is not None]))
            last_prog = mapped[lis_idx[-1]] if lis_idx else -1
            cands.append((key, length, coverage, last_prog))
        cands.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        trip_candidates[tid] = cands[:K] if cands else [] """

    # Ranking por adhesión sostenida (geometría) ===
    geoms = prepare_route_geoms(stations_ord) 

    rows2 = []
    for tid, g in df.groupby("trip_id", sort=False):
        g = g.sort_values("Fecha")
        unique_stations_visited = g["estacion_cercana"].dropna().astype(str).unique().tolist()
        lat = g["Latitud"].to_numpy(dtype=float)
        lon = g["Longitud"].to_numpy(dtype=float)
        ts  = g["Fecha"]                      # Series alineada con lat/lon
        idx = g.index                         # índices reales en df
        
        # Preselección de rutas a probar:
        cand_list = None

        if 'SAN RAFAEL' in unique_stations_visited or 'PARAÍSO' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 18 - B'.casefold()]
            
        elif 'ATLÁNTIDA' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 18 - A'.casefold()] + [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 18 - B'.casefold()]
            
        elif 'CEJUSA ANDÉN SUR' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 7'.casefold()]
            
        elif 'JOCOTENANGO' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 2'.casefold()]
            
        elif 'CENTRO ZONA 6' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 6'.casefold()]
            
        elif 'EXPOSICIÓN' in unique_stations_visited:
            cand_list = [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 13 - A'.casefold()] + [ (key, 0, 0, 0) for key in geoms.keys() if key[0].strip().casefold() == 'Linea 13 - B'.casefold()]
            
            
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
                    if (r["t_start"] < cur["t_start"]) and (r["score"] >= 0.5*cur["score"]):
                        best_by_line[key] = r
                        
            best = max(best_by_line.values(), key=lambda r: r["score"])
            rows2.append(best)
        else:
            rows2.append({"trip_id": tid, "LINEA": None, "DIR": None,
                        "t_start": None, "t_start_ts": None, "idx_start": None,
                        "t_end": None, "t_end_ts": None, "idx_end": None,
                        "progress_m": 0.0, "mean_dev_m": np.nan,
                        "frac_in": 0.0, "dur_pts": 0, "score": -1e9})

    route_scores_df = pd.DataFrame(rows2)

    # Ordenar resultados por trip_id numéricamente
    route_scores_df["trip_id_num"] = pd.to_numeric(route_scores_df["trip_id"], errors="coerce")
    route_scores_df = route_scores_df.sort_values("trip_id_num").drop(columns=["trip_id_num"])

    # Guardar resultados
    OUT_CSV = f"../data_with_features/{UNIT}/{UNIT}_trip_routes.csv"

    # Crear carpeta si no existe
    import os
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    route_scores_df.to_csv(OUT_CSV, index=False)
    
    # Guardar trips con t_start > 300 (muy probables de ser multi-línea)
    multiline_trips = route_scores_df[route_scores_df["t_start"] > 300]
    OUT_MULTI_CSV = f"../data_with_features/{UNIT}/{UNIT}_multiline_trips.csv"
    multiline_trips.to_csv(OUT_MULTI_CSV, index=False)

# ----------- EJECUCIÓN PRINCIPAL -----------

if __name__ == "__main__":
    
    def parse_pos(s):
        a, b = [float(t.strip()) for t in str(s).split(",")]
        return pd.Series({"LAT": a, "LON": b})
    
    def discover_units(input_dir):
        units = []
        for path in glob.glob(os.path.join(input_dir, "*/*_clean_trips.csv")):
            base = os.path.basename(path)  # "u001_clean_trips.csv"
            unit = base.replace("_clean_trips.csv", "")  # "u001"
            units.append(unit)
        return sorted(units)
    
    input_dir = "../clean_data/"
    units = discover_units(input_dir)
    
    STATIONS_XLS= "../data/Estaciones_ordenadas_with_pos.xlsx"

    # Carga estaciones (Excel): usa POSICIÓN "lat, lon" y ordena por ORDEN.
    stations_ord = pd.read_excel(STATIONS_XLS)
    stations_ord[["LAT", "LON"]] = stations_ord["POSICIÓN"].apply(parse_pos)
    stations_ord = stations_ord.sort_values("ORDEN")

    for unit in units:
        print(f'Procesando unidad {unit}')
        
        # Si ya existe el archivo de salida, omitir
        out_csv = f"../data_with_features/{unit}/{unit}_trip_routes.csv"
        if os.path.exists(out_csv):
            print(f'El archivo {out_csv} ya existe. Se omite el procesamiento.')
            continue

        add_adhesion_feature(unit, stations_ord)