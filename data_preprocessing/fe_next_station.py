'''Script de Ingeniería de características para asignar la siguiente estación en cada punto de cada viaje,
basado en la adherencia a la línea.'''

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
 
def plot_projection_debug(px, py, route, debug, save_path=None):
    """
    Plot: route polyline, candidate segment projections, chosen projection, and the point.
    One chart only, no explicit colors.
    """
    rx, ry = route["rx"], route["ry"]
    fig, ax = plt.subplots(figsize=(7, 6))

    # route polyline
    ax.plot(rx, ry, linewidth=2)

    # point
    ax.scatter([px], [py], marker="x", s=70)

    # candidate projections and lines from point to projection
    cand = debug["cand"]
    projx = debug["projx"]
    projy = debug["projy"]
    for cx, cy in zip(projx, projy):
        ax.plot([px, cx], [py, cy], linewidth=1, alpha=0.7)

    # chosen projection
    j = debug["best_idx_in_cand"]
    ax.scatter([projx[j]], [projy[j]], s=90)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Proyección del punto sobre la polilínea (con candidatos)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

def window_k_for(route, target_m=200.0, k_min=8):
    rx, ry = route["rx"], route["ry"]
    seg_len = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2)
    avg = max(1.0, float(np.mean(seg_len)))  # m/segmento
    return int(max(k_min, np.ceil(target_m / avg)))

def project_point_to_polyline(px, py, route, prev_seg_idx=None, prev_px=None, prev_py=None, prev_s=None, k=25,
                              max_step_fwd=200.0, back_tolerance=20.0,
                              lam_back=1e-2, lam_fwd=1e-3, lam_idx=1e-4, handoff_thresh=50.0, switch_direction_in_same_station=False, current_global_index=None):
    
    rx, ry, rc = route["rx"], route["ry"], route["route_cum"]
    dx, dy = np.diff(rx), np.diff(ry)
    ax, ay = rx[:-1], ry[:-1]
    seg2 = dx*dx + dy*dy
    seg_len = np.sqrt(seg2, where=seg2>0, out=np.zeros_like(seg2))

    # Ventana local
    if prev_seg_idx is not None:
        i0 = max(0, prev_seg_idx - k)
        i1 = min(len(dx)-1, prev_seg_idx + k)
        cand = np.arange(i0, i1+1)
    else:
        cand = np.arange(len(dx))
    if cand.size == 0:
        return float('inf'), float('nan'), None, False

    pax = px - ax[cand];  pay = py - ay[cand]
    t = np.divide(pax*dx[cand] + pay*dy[cand], seg2[cand],
                  out=np.zeros_like(seg2[cand]), where=seg2[cand]>0)
    t = np.clip(t, 0, 1)
    projx = ax[cand] + t*dx[cand]
    projy = ay[cand] + t*dy[cand]
    d2 = (px - projx)**2 + (py - projy)**2
    s_cand = rc[cand] + t*seg_len[cand]

    # Costo total
    cost = d2.copy()
    if prev_s is not None:
        back = np.maximum(0.0, prev_s - s_cand - back_tolerance)
        cost += lam_back * back*back
        fwd_excess = np.maximum(0.0, s_cand - (prev_s + max_step_fwd))
        cost += lam_fwd * fwd_excess*fwd_excess
    if prev_seg_idx is not None:
        cost += lam_idx * (cand - prev_seg_idx)**2
        
    j_loc = int(np.argmin(cost))
    j = int(cand[j_loc])
    s = float(s_cand[j_loc])
    d = float(np.sqrt(d2[j_loc]))
    went_back = (prev_s is not None and s < prev_s)
        
    # --- Determinar si hay que cambiar de dirección ---
    
    switch_direction = False
    vector_in_opposite_dir = False
    
    # Si ya se detectó el último índice, cambiar
    
    # Poco fiable en cambios de dirección en la misma estación: Si cambia de dirección a la primera instancia de la última estación de la dirección actual (muy temprano),
    # puede que no haya empezado el trayecto de la polilínea de la siguiente dirección y en su lugar,
    # regresará como mejor candidato la estación "más cercana", y, como en términos de "s"
    # pareciera que ya pasó la primera estación, se la saltará.
    """ if prev_seg_idx == len(dx) - 1:
        switch_direction = True
        print("Cambio de dirección por índice menor") """
        
    # CAMBIO DE DIRECCIÓN EN ESTACIÓN DIFERENTE: Si la proyección más cercana está en la última estación de la ruta (último segmento)
    # y la distancia a esta es menor que un umbral, cambiar
    if prev_seg_idx is not None and prev_seg_idx == len(dx) - 1 and switch_direction_in_same_station == False:
        meters_to_end = route["route_cum"][-1] - s
        in_end_window = meters_to_end <= 50.0
        if j == len(dx) - 1 and in_end_window:
            switch_direction = True
            #print("Cambio de dirección por proyección en última estación en punto ", current_global_index)

    # CAMBIO DE DIRECCIÓN EN MISMA ESTACIÓN: Producto punto del último vector de moviento con el vector del último segmento
    if prev_px is not None and prev_py is not None and prev_seg_idx is not None and prev_seg_idx == len(dx) - 1 and switch_direction_in_same_station == True:
        
        # vector de movimiento
        v_move = np.array([px - prev_px, py - prev_py])
        nv = np.linalg.norm(v_move)
        if nv > 1e-6:
            v_move /= nv
        else:
            v_move[:] = 0.0
            
        # vector del último segmento
        v_seg = np.array([dx[-1], dy[-1]])
        nseg = np.linalg.norm(v_seg)
        if nseg > 1e-6:
            v_seg /= nseg
        else:
            v_seg[:] = 0.0
            
        # producto punto
        dp = np.dot(v_move, v_seg)
        
        # Decisión: Producto punto negativo y la distancia a la proyección anterior es mayor que un umbral (para evitar ruido)
        if dp < 0.0: 
            vector_in_opposite_dir = True
        
    # Segundo, si la proyección más cercana está muy atrás, cambiar
    if prev_seg_idx is not None and switch_direction == False:
        j_loc_min = int(np.argmin(d2))
        j_min = int(cand[j_loc_min])
        if j_min < prev_seg_idx - 3:
            switch_direction = True
            #print("Cambio de dirección por proyección atrás en punto ", current_global_index)
    
    debug = {
        "cand": cand,
        "t": t,
        "projx": projx,
        "projy": projy,
        "d2": d2,
        "s_cand": s_cand,
        "best_idx_in_cand": j_loc,
        "best_s": s,
        "best_d": d,
        "best_seg": j,
        "cost": cost,
        "px": px,
        "py": py
    }

    return d, s, j, switch_direction, vector_in_opposite_dir, went_back, debug

# ---- Helpers específicos para continuidad IDA/VUELTA ----
def opposite_dir(d):
    d = str(d).strip().upper()
    if d == "IDA": return "VUELTA"
    if d == "VUELTA": return "IDA"
    return d
    
def project_many_on_route(route, lat_arr, lon_arr,
                          max_step_fwd=200.0, back_tolerance=20.0,
                          lam_back=1e-2, lam_fwd=1e-3, lam_idx=1e-4, switch_direction_in_same_station=False, current_global_index=None):
    
    k = 50
    
    px, py = ll_to_xy_m(lat_arr, lon_arr, route["lat0"], route["lon0"])
    s_list, d_list = [], []
    debug_list = []
    prev_seg, prev_s, prev_x, prev_y = None, None, None, None

    # Manejar cambio de dirección
    vectors_in_opposite_dir = 0
    switch_direction = False
    cut_n = 0  # cuántos puntos logramos proyectar

    for idx, (x, y) in enumerate(zip(px, py)):
        d, s, prev_seg, sw, vector_in_opposite_dir, went_back, debug = project_point_to_polyline(
            x, y, route, prev_seg_idx=prev_seg, prev_px=prev_x, prev_py=prev_y, prev_s=prev_s, k=k,
            max_step_fwd=max_step_fwd, back_tolerance=back_tolerance,
            lam_back=lam_back, lam_fwd=lam_fwd, lam_idx=lam_idx, switch_direction_in_same_station=switch_direction_in_same_station, current_global_index=current_global_index + idx + 1
        )

        # histéresis anti-retroceso
        if prev_s is not None and s < prev_s:
            s = max(prev_s - back_tolerance, s)

        s_list.append(s)
        d_list.append(d)
        debug_list.append(debug)
        prev_s = s
        prev_x = x
        prev_y = y
        cut_n = idx + 1

        if sw:
            switch_direction = True
            break  # detenemos aquí para que el caller procese este tramo y luego cambie de dir
        
        if vector_in_opposite_dir:
            vectors_in_opposite_dir += 1
        if vectors_in_opposite_dir >= 3:
            #print(f"Cambio de dirección por vector en punto {current_global_index + idx + 1}")
            switch_direction = True
            break  # detenemos aquí para que el caller procese este tramo y luego cambie de dir

    # done=True si consumimos TODO lat_arr/lon_arr (no hubo switch)
    done = (cut_n == len(lat_arr))
    return np.array(s_list, float), np.array(d_list, float), switch_direction, cut_n, done, debug_list

def next_station_on_route(route, stations_df, s_arr):
    """
    Para cada s en s_arr (posición sobre la ruta), devuelve:
    - nombre de próxima estación
    - delta de distancia hasta esa estación
    """
    st = stations_df.sort_values("s_est")
    s_est = st["s_est"].to_numpy(float)
    names = st["station"].tolist()

    idxs = np.searchsorted(s_est, s_arr, side="right") # Ordena las distancias según su cercanía a las estaciones
    
    # Si la distancia es exactamente igual a una estación, se toma esta como la siguiente, ya que puede que no haya empezado
    # el trayecto de la polilínea de la siguiente dirección.
    
    # Por tanto, si s coincide exactamente con la estación previa, retrocede 1
    """ for i, s in enumerate(s_arr):
        j = idxs[i]
        if j > 0 and np.isclose(s, s_est[j-1], atol=1e-6):
            idxs[i] = j - 1 """
    
    # Nombres y deltas
    prox_names = []
    deltas = []
    for s, idx in zip(s_arr, idxs):
        
        if idx >= len(s_est):
            # fin de ruta
            if route["is_circular"]:
                prox_names.append(names[0])
                deltas.append(route["length_m"] - s + s_est[0])
            else:
                prox_names.append(names[-1])
                deltas.append(max(0.0, s_est[-1] - s))  # clamp a 0
        else:
            prox_names.append(names[idx])
            deltas.append(s_est[idx] - s)

        
    return prox_names, np.array(deltas, float)

def end_of_dir_mask(s_arr, last_s_est, eps=20.0):
    """Marca puntos que ya están ~al final de la dirección actual."""
    return s_arr >= (last_s_est - eps)

# Para debuggear proyecciones
def build_folium_trace_map(route, lat0, lon0, debug_list, station_names=None,
                           html_path="/mnt/data/trace_projection_debug.html",
                           show_candidates=True, cluster=False):
    """
    Creates a single HTML with:
      - Route polyline + stations
      - For each point: marker of measurement, line to best projection, and (optionally) candidate projections.
    """
    rx, ry = route["rx"], route["ry"]
    # Convert full route to lat/lon for folium
    lats, lons = xy_to_ll_m(rx, ry, lat0, lon0)
    center = [float(np.mean(lats)), float(np.mean(lons))]
    m = folium.Map(location=center, zoom_start=14, control_scale=True)

    # Layer control container
    fg_route = folium.FeatureGroup(name="Ruta", show=True)
    # Polyline for route
    coords = [[float(la), float(lo)] for la, lo in zip(lats, lons)]
    folium.PolyLine(coords, weight=6, opacity=0.8, tooltip="Ruta").add_to(fg_route)
    fg_route.add_to(m)

    # Stations/vertices
    for i, (la, lo) in enumerate(zip(lats, lons)):
        title = f"Vértice {i}"
        if station_names and i < len(station_names):
            title += f" — {station_names[i]}"
        folium.CircleMarker([float(la), float(lo)], radius=5, tooltip=title).add_to(fg_route)

    # Points layer
    fg_points = folium.FeatureGroup(name="Puntos proyectados", show=True)
    fg_candidates = folium.FeatureGroup(name="Candidatos (proyecciones)", show=False)

    # Add each point and its chosen projection
    for idx, dbg in enumerate(debug_list):
        # Original measurement (px, py) to lat/lon
        p_lat, p_lon = xy_to_ll_m(dbg["px"], dbg["py"], lat0, lon0)
        # Best projection
        j = dbg["best_idx_in_cand"]
        if j is None:
            continue
        bx, by = dbg["projx"][j], dbg["projy"][j]
        b_lat, b_lon = xy_to_ll_m(bx, by, lat0, lon0)

        # Marker for point
        folium.CircleMarker(
            [p_lat, p_lon],
            radius=3,
            tooltip=f"Punto {idx}",
            popup=f"Punto {idx}<br>best_d={dbg['best_d']:.2f} m<br>best_s={dbg['best_s']:.2f} m<br>seg={dbg['best_seg']}",
        ).add_to(fg_points)
        # Line to best projection
        folium.PolyLine([[p_lat, p_lon], [b_lat, b_lon]], weight=2, opacity=0.8).add_to(fg_points)

        # Optional candidate projections
        if show_candidates and len(dbg.get("cand", [])) > 0:
            for cx, cy, s in zip(dbg["projx"], dbg["projy"], dbg["s_cand"]):
                cla, clo = xy_to_ll_m(cx, cy, lat0, lon0)
                folium.CircleMarker([cla, clo], radius=2, opacity=0.9, tooltip=f"s={s:.2f}").add_to(fg_candidates)

    fg_points.add_to(m)
    if show_candidates:
        fg_candidates.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(html_path)
    return html_path

# =========================
# Próxima estación teórica con continuidad
# =========================
def compute_next_for_segment(
    segment: pd.DataFrame,
    segment_stable_start_row: pd.Series,   # fila de trip_routes (tiene LINEA, DIR, idx_start, etc.)
    geoms: dict,
    stations_by_key: dict,
    win_confirm_pts: int = 8,           # ~ventana corta para confirmar VUELTA
    eps_end_m: float = 20.0,            # tolerancia para considerar "final de ruta"
    dist_margin: float = 20.0,
    min_progress_confirm: float = 200.0,
    dist_thresh: float = 200.0,
    frac_within: float = 0.75,
    debug_html_path: str = None
) -> pd.DataFrame:
    """
    Calcula s_m, dist_m, proxima_est_teorica y dist_a_prox_m para un trip,
    usando la DIR inferida a partir de idx_start; permite cambiar a la dir opuesta
    cuando el bus llega al final y la ventana confirma el cambio.

    - NO mezcla estaciones de IDA y VUELTA en un mismo 's_est' (cada DIR tiene su marco).
    - Antes de idx_start, rellena con la primera próxima estación calculada en idx_start.
    """
    out = segment.copy()

    # 1) Desde dónde es válido este trip:
    stable_start_index = int(segment_stable_start_row.get("idx_start", 0)) if segment_stable_start_row is not None else 0
    if stable_start_index < 0 or stable_start_index >= len(out):
        stable_start_index = 0

    work = out.iloc[stable_start_index:].copy().reset_index(drop=False)  # guarda índice original en 'index'

    # 2) Ruta activa inicial (según inferencia)
    linea = segment_stable_start_row.get("LINEA")
    dir0  = segment_stable_start_row.get("DIR")

    if pd.isna(linea) or pd.isna(dir0):
        # Sin línea o dir inferidas, devuelve NaN/None
        out["s_m"] = np.nan
        out["dist_m"] = np.nan
        out["dist_a_prox_m"] = np.nan
        out["proxima_est_teorica"] = pd.Series(index=out.index, dtype="object")
        out["DIR"] = pd.Series(index=out.index, dtype="object")
        return out
    
    # Para líneas con IDA/VUELTA
    dir1 = None
    route1 = None
    dir0_last_station_equals_first_of_dir1 = False
    dir1_last_station_equals_first_of_dir0 = False
    
    if dir0 in ["IDA", "VUELTA"]:
        dir1 = opposite_dir(dir0)
        
        # Verifica si la última estación de dir0 es la primera de dir1
        seq0 = stations_by_key.get((linea, dir0), pd.DataFrame()).sort_values("s_est")["station"].tolist()
        seq1 = stations_by_key.get((linea, dir1), pd.DataFrame()).sort_values("s_est")["station"].tolist()
        if seq0 and seq1 and seq0[-1] == seq1[0]:
            dir0_last_station_equals_first_of_dir1 = True
            #print(f"Línea {linea}: última estación de {dir0} es la primera de {dir1}")
            
        # Verifica si la última estación de dir1 es la primera de dir0
        if seq1 and seq0 and seq1[-1] == seq0[0]:
            dir1_last_station_equals_first_of_dir0 = True
            #print(f"Línea {linea}: última estación de {dir1} es la primera de {dir0}")

    # --- setup rutas/estaciones actuales ---
    key_dir0 = (linea, dir0)
    route_0 = geoms.get(key_dir0)
    if route_0 is None or key_dir0 not in stations_by_key:
        out.loc[:, ["s_m","dist_m","proxima_est_teorica","dist_a_prox_m"]] = [np.nan, np.nan, None, np.nan]
        return out

    dir1 = opposite_dir(dir0) if dir0 in ["IDA","VUELTA"] else None
    route_1 = geoms.get((linea, dir1)) if dir1 else None

    def pick_route_and_st(d):
        r = geoms.get((linea, d))
        st = stations_by_key.get((linea, d), pd.DataFrame()).sort_values("s_est")
        return r, st

    current_dir = dir0
    current_route, st_cur = pick_route_and_st(current_dir)
    current_switch_dir_in_same_station = dir0_last_station_equals_first_of_dir1  # si el cambio de dir es en la misma estación (última == primera)

    # arrays del tramo válido
    latv = work["Latitud"].to_numpy(float)
    lonv = work["Longitud"].to_numpy(float)
    n = len(latv)

    # prealoca columnas en work
    work["s_m"] = pd.Series(index=work.index, dtype="float64")
    work["dist_m"] = pd.Series(index=work.index, dtype="float64")
    work["dist_a_prox_m"] = pd.Series(index=work.index, dtype="float64")

    work["proxima_est_teorica"] = pd.Series(index=work.index, dtype="object")
    work["DIR"] = pd.Series(index=work.index, dtype="object")

    p = 0  # puntero posicional
    safe_guard = 0
    max_iters = 2 * n + 10  # por seguridad
    
    debug_dir_0 = []
    debug_dir_1 = []
    
    current_global_index = stable_start_index

    while p < n and safe_guard < max_iters:
        safe_guard += 1

        s_arr, d_arr, switched, cut_n, done, debug = project_many_on_route(
            current_route, latv[p:], lonv[p:],
            max_step_fwd=1200.0, back_tolerance=20.0,
            lam_back=1e-2, lam_fwd=1e-3, lam_idx=1e-4, switch_direction_in_same_station=current_switch_dir_in_same_station, current_global_index=current_global_index
        )

        if cut_n == 0:
            # nada proyectado: evita loop
            break
        
        if current_dir == dir0:
            debug_dir_0.extend(debug)
        else:
            debug_dir_1.extend(debug)

        # Próximas estaciones en el tramo proyectado
        prox_names, deltas = next_station_on_route(current_route, st_cur, s_arr)

        # Escribir SOLO el tramo [p : p+cut_n)
        idx_slice = work.index[p:p+cut_n]
        work.loc[idx_slice, "s_m"] = s_arr
        work.loc[idx_slice, "dist_m"] = d_arr
        work.loc[idx_slice, "proxima_est_teorica"] = [str(x) if x is not None else None for x in prox_names]
        work.loc[idx_slice, "dist_a_prox_m"] = deltas
        work.loc[idx_slice, "DIR"] = current_dir

        p += cut_n
        current_global_index += cut_n

        if switched and dir1:
            
            # conmutar dir y sus estaciones
            current_dir = dir1 if current_dir == dir0 else dir0
            current_route, st_cur = pick_route_and_st(current_dir)
            current_switch_dir_in_same_station = (current_dir == dir0 and dir0_last_station_equals_first_of_dir1) or (current_dir == dir1 and dir1_last_station_equals_first_of_dir0)
            
            # continúa el while con la nueva dirección
            
        elif done:
            break

    # Vuelca 'work' al 'out' respetando el índice original guardado en 'index'
    
    cols = ["s_m","dist_m","proxima_est_teorica","dist_a_prox_m","DIR"]
    for c in cols:
        out.loc[work["index"], c] = work[c].values  # o sin .values: out.loc[work["index"], c] = work[c]

    # Relleno previo a stable_start_index
    if stable_start_index > 0 and len(work) > 0:
        first_est = work["proxima_est_teorica"].iloc[0]
        first_dst = work["dist_a_prox_m"].iloc[0]
        out.loc[out.index[:stable_start_index], "proxima_est_teorica"] = first_est
        out.loc[out.index[:stable_start_index], "dist_a_prox_m"] = first_dst
        out.loc[out.index[:stable_start_index], "DIR"] = work["DIR"].iloc[0]
        
    # Debug: guardar mapa de proyecciones para ida y vuelta
    if debug_html_path and len(debug_dir_0) > 0:
        
        # ruta ida
        debug_html_path = Path(debug_html_path)
        debug_html_path.parent.mkdir(parents=True, exist_ok=True)
        debug_html_path_ida = debug_html_path.with_name(debug_html_path.stem + "_ida.html")
        build_folium_trace_map(
            route_0, route_0["lat0"], route_0["lon0"], 
            [dbg for dbg in debug_dir_0 if dbg["best_seg"] is not None],
            station_names=st_cur["station"].tolist(),
            html_path=str(debug_html_path_ida),
            show_candidates=False,
            cluster=False
        )
        
        # ruta vuelta (si aplica)
        if route_1:
            debug_html_path_vta = debug_html_path.with_name(debug_html_path.stem + "_vuelta.html")
            build_folium_trace_map(
                route_1, route_1["lat0"], route_1["lon0"], 
                [dbg for dbg in debug_dir_1 if dbg["best_seg"] is not None],
                station_names=stations_by_key.get((linea, dir1), pd.DataFrame())["station"].tolist(),
                html_path=str(debug_html_path_vta),
                show_candidates=False,
                cluster=False
            )
            print(f"Mapa de proyecciones guardado en {debug_html_path_ida} y {debug_html_path_vta}")
        else:
            print(f"Mapa de proyecciones guardado en {debug_html_path_ida}")

    # 5) Copia resultados al DataFrame original
    #    (recuerda que 'work' tiene la columna 'index' con el índice original)
    out.loc[work["index"], ["s_m","dist_m","proxima_est_teorica","dist_a_prox_m","DIR"]] = work[["s_m","dist_m","proxima_est_teorica","dist_a_prox_m","DIR"]].values

     # 6) Antes del índice estable:
    #    - fija la próxima estación igual a la del primer punto estable,
    #    - recalcula la DISTANCIA a esa estación con la posición real de cada punto previo.
    if stable_start_index > 0 and len(work) > 0:
        # estación/dir confirmadas en el punto estable
        first_est = str(work["proxima_est_teorica"].iloc[0]) if pd.notna(work["proxima_est_teorica"].iloc[0]) else None
        start_dir = str(dir0)

        # si falta info, no hacemos nada
        if first_est is not None and (linea, start_dir) in geoms and (linea, start_dir) in stations_by_key:
            route_start = geoms[(linea, start_dir)]
            st_start = stations_by_key[(linea, start_dir)].sort_values("s_est")

            # s_est de la estación objetivo
            match = st_start.loc[st_start["station"] == first_est, "s_est"]
            if not match.empty:
                s_target = float(match.iloc[0])

                # subset de puntos previos
                pre_idx = out.index[:stable_start_index]
                lat_pre = out.loc[pre_idx, "Latitud"].astype(float).to_numpy()
                lon_pre = out.loc[pre_idx, "Longitud"].astype(float).to_numpy()

                # proyecta puntos previos a la ruta de inicio (sin permitir switch)
                s_pre, d_pre, _sw, cut_n, _done, _dbg = project_many_on_route(
                    route_start, lat_pre, lon_pre,
                    max_step_fwd=1200.0, back_tolerance=20.0,
                    lam_back=1e-2, lam_fwd=1e-3, lam_idx=1e-4,
                    switch_direction_in_same_station=False,
                    current_global_index=stable_start_index - len(pre_idx)
                )

                # asegura longitud (por si cut_n < len)
                s_pre = np.pad(s_pre, (0, max(0, len(pre_idx) - len(s_pre))), constant_values=np.nan)
                d_pre = np.pad(d_pre, (0, max(0, len(pre_idx) - len(d_pre))), constant_values=np.nan)

                # distancia a la estación objetivo (clamp mínimo 0 si ya la pasó numéricamente)
                dist_to_target = np.maximum(0.0, s_target - s_pre)

                # escribe resultados previos
                out.loc[pre_idx, "proxima_est_teorica"] = first_est
                out.loc[pre_idx, "DIR"] = start_dir
                out.loc[pre_idx, "s_m"] = s_pre
                out.loc[pre_idx, "dist_m"] = d_pre
                out.loc[pre_idx, "dist_a_prox_m"] = dist_to_target
            else:
                # si no se encuentra la estación por nombre, al menos fija nombre/DIR
                out.loc[out.index[:stable_start_index], "proxima_est_teorica"] = first_est
                out.loc[out.index[:stable_start_index], "DIR"] = start_dir

    return out

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
    stations_by_key = {}

    for (linea, d), g in df.sort_values("order").groupby(["name","dir"], sort=False):
        g = g.copy()

        # 1) Construye la polilínea con los vértices en orden

        latv = g["lat"].astype(float).to_numpy()
        lonv = g["lon"].astype(float).to_numpy()
        lat0, lon0 = float(latv.mean()), float(lonv.mean())

        rx, ry = ll_to_xy_m(latv, lonv, lat0, lon0)
        route_cum = cumulative_distances(rx, ry)

        route = dict(
            rx=rx, ry=ry, route_cum=route_cum,
            lat0=lat0, lon0=lon0,
            length_m=float(route_cum[-1]),
            is_circular=False  # si alguna línea es circular, cámbialo aquí
        )
        geoms[(linea, d)] = route

        # 2) Estaciones sobre esta polilínea
        st = g[g["kind"].str.lower()=="station"].copy()
        rows = []
        for _, r in st.iterrows():
            s_est = station_s_on_route(route, float(r["lat"]), float(r["lon"]))
            rows.append({
                "station": (r["station"] if "station" in r and pd.notna(r["station"]) else None),
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "s_est": float(s_est)
            })
        stations_by_key[(linea, d)] = pd.DataFrame(rows).sort_values("s_est").reset_index(drop=True)

    return geoms, stations_by_key

def station_s_on_route(route, lat, lon):
    x, y = ll_to_xy_m(lat, lon, route["lat0"], route["lon0"])
    rx, ry, rc = route["rx"], route["ry"], route["route_cum"]
    dx, dy = np.diff(rx), np.diff(ry)
    ax, ay = rx[:-1], ry[:-1]
    seg2 = dx*dx + dy*dy
    # t de proyección, recortado a [0,1]
    t = np.divide((x - ax)*dx + (y - ay)*dy, seg2,
                  out=np.zeros_like(seg2), where=seg2>0)
    t = np.clip(t, 0, 1)
    projx = ax + t*dx
    projy = ay + t*dy
    d2 = (x - projx)**2 + (y - projy)**2
    j = int(np.argmin(d2))                     # mejor segmento
    s = float(rc[j] + t[j]*np.sqrt(seg2[j]))   # s sobre la ruta
    
    return s


# --- Ejecución principal ---
# Cargar geometrías de todas las líneas

LINES = ['Linea_1', 'Linea_2', 'Linea_6', 'Linea_7', 'Linea_12', 'Linea_13-A', 'Linea_13-B', 'Linea_18-A', 'Linea_18-B']

geoms = {}
stations_by_key = {}

for line in LINES:
    path = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_preprocessing\\complete_lines\\{line}_with_direction.csv"
    g, s = load_line_polyline_csv(path)
    geoms.update(g)
    stations_by_key.update(s)
    
# --- Ejecución principal ---

def process_unit_next_station(unit):
    
    print(f'-- Procesando unidad {unit} --')
    
    all_results = []  # acumular todos los segmentos procesados
    
    UNIT = unit
    CLEAN_TRIPS_CSV = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\clean_data\\{UNIT}\\{UNIT}_clean_trips.csv"
    TRIPS_VITERBI = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{UNIT}\\{UNIT}_trips_with_viterbi.csv"
    DEBUG_HTML_PATH = None

    df = pd.read_csv(CLEAN_TRIPS_CSV, dtype={"trip_id": int})
    
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.sort_values(["trip_id","Fecha"])
    trips_viterbi_df = pd.read_csv(TRIPS_VITERBI, dtype={"trip_id": int})
    
    # Obtener los índices "estables" de los bloques
    stable_indices = trips_viterbi_df[["idx_start", "idx_end"]].drop_duplicates().reset_index(drop=True)
    
    for trip_id, trip in df.groupby("trip_id", sort=False):
        segs = trips_viterbi_df[trips_viterbi_df["trip_id"] == trip_id].copy()
        if segs.empty:
            print(f"Trip {trip_id} sin segmentos en {TRIPS_VITERBI}, se omite.")
            continue

        # Asegurar tipos enteros
        for c in ["idx_start", "idx_end", "t_start", "t_end", "dur_pts"]:
            if c in segs.columns:
                segs[c] = segs[c].astype(float).astype(int)

        # Orden lógico de bloques
        segs = segs.sort_values(["t_start", "idx_start"])

        # Procesar bloque por bloque
        for blk_id, seg in enumerate(segs.itertuples(index=False), start=1):
            linea = seg.LINEA
            dir_init = getattr(seg, "DIR_init", None)
            if pd.isna(linea) or linea not in LINES:
                print(f"• Bloque {blk_id}: línea inválida '{linea}', se salta.")
                continue
            if dir_init is None:
                print(f"• Bloque {blk_id}: sin DIR_init, se salta.")
                continue

            i0 = int(seg.idx_start) # Inicio del bloque
            
            # El fin del bloque es el último índice del trip
            i1 = trip.index[-1]

            trip_index_start = seg.t_start

            # subset del trip original para este bloque
            # Esto se hace para viajes multilínea
            seg_df = df.loc[i0:i1].copy()
            
            if seg_df.empty:
                print(f"• Bloque {blk_id}: ventana {i0}-{i1} vacía, se salta.")
                continue

            # construir una “fila de estado inicial” con el nombre de columna que espera compute_next_for_trip
            start_row = pd.Series({
                "LINEA": linea,
                "DIR": dir_init,
                "idx_start": trip_index_start
            })

            res = compute_next_for_segment(
                segment=seg_df,
                segment_stable_start_row=start_row,
                geoms=geoms,
                stations_by_key=stations_by_key,
                win_confirm_pts=8,
                eps_end_m=20.0,
                dist_margin=20.0,
                min_progress_confirm=200.0,
                dist_thresh=200.0,
                frac_within=0.75,
                debug_html_path=DEBUG_HTML_PATH
            )

            # anotar metadatos del bloque
            res = res.copy()
            res["trip_id"] = trip_id
            res["LINEA"] = linea
            res["DIR_init"] = dir_init
            res["block_id"] = blk_id
            res["idx_start_blk"] = i0
            res["idx_end_blk"] = i1

            all_results.append(res)

    # Guardar concatenado
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = f"D:\\2025\\UVG\\Tesis\\repos\\backend\\data_with_features\\{UNIT}\\{UNIT}_trips_with_next_station.csv"
        import os; os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_df.to_csv(out_path, index=False)
        print(f"Resultados guardados en: {out_path}")
    else:
        print("No hubo resultados para guardar.")
        
if __name__ == "__main__":
    """ print('=== Iniciando inferencia de próxima estación ===')
    # Encontrar todas las unidades (carpetas en data with features)
    DATA_WITH_FEATURES_DIR = Path("D:/2025/UVG/Tesis/repos/backend/data_with_features")
    if not DATA_WITH_FEATURES_DIR.exists():
        print(f"No existe el directorio {DATA_WITH_FEATURES_DIR}. Fin.")
    units = [p.name for p in DATA_WITH_FEATURES_DIR.iterdir() if p.is_dir() and p.name != "maps"]
    if not units:
        print(f"No se encontraron unidades en {DATA_WITH_FEATURES_DIR}. Fin.") """
    units = ['u049', 'u050', 'u051', 'u052', 'u053', 'u055', 'u056', 'u057', 'u059', 'u060', 'u061', 'u062', 'u063', 'u064', 'u066', 'u067', 'u068', 'u069', 'u070', 'u071', 'u074', 'u075', 'u086', 'u087', 'u088', 'u089', 'u090', 'u091', 'u092', 'u093', 'u094', 'u095', 'u096', 'u097', 'u098', 'u099', 'u100', 'u101', 'u102', 'u104', 'u105', 'u106', 'u107', 'u110', 'u111', 'u112', 'u113', 'u114', 'u115', 'u116', 'u117', 'u118', 'u119', 'u120', 'u121', 'u122', 'u123', 'u124', 'u125', 'u127', 'u128', 'u129', 'u130', 'u131', 'u132', 'u133', 'u134', 'u135', 'u136', 'u137', 'u138', 'u139', 'u140', 'u141', 'u142', 'u144', 'u145', 'u146', 'u148', 'u149', 'u150', 'u151', 'u152', 'u153', 'u154', 'u155', 'u156', 'u157', 'u158', 'u159', 'u160', 'u161', 'u201', 'u203', 'u204', 'u205', 'u206', 'u207', 'u208', 'u210', 'u211', 'u212', 'u213', 'u214', 'u215', 'u216', 'u217', 'u218', 'u219', 'u220', 'u221', 'u222', 'u223', 'u224', 'u225', 'u226', 'u227', 'u228', 'u231', 'u232', 'u233', 'u234', 'u235', 'u236', 'u237', 'u238', 'u239', 'u240', 'u241', 'u242', 'u301', 'u302', 'u303', 'u304', 'u305', 'u306', 'u307', 'u308', 'u309', 'u310', 'u401', 'u402', 'uBC232', 'uBC322', 'uBI002', 'uBI003', 'uBI004', 'uBI005', 'uBI006', 'uBI007', 'uBI008', 'uBI009', 'uBI010', 'uBI011', 'uBI012', 'uBI013']
    for unit in units:
        process_unit_next_station(unit)
