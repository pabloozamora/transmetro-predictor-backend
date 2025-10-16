import pandas as pd
import numpy as np
import math

# --------- UTILS ---------
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

def parse_pos(s):
        a, b = [float(t.strip()) for t in str(s).split(",")]
        return pd.Series({"LAT": a, "LON": b})

# -------------------------

def point_d_and_s_on_route(route, lat, lon):
    x, y = ll_to_xy_m(np.array([lat]), np.array([lon]), route["lat0"], route["lon0"])
    rx, ry, rc = route["rx"], route["ry"], route["route_cum"]
    dx, dy = np.diff(rx), np.diff(ry)
    ax, ay = rx[:-1], ry[:-1]
    seg2 = dx*dx + dy*dy

    t = np.divide((x - ax)*dx + (y - ay)*dy, seg2, out=np.zeros_like(seg2), where=seg2>0)
    t = np.clip(t, 0, 1)
    projx = ax + t*dx
    projy = ay + t*dy
    d2 = (x - projx)**2 + (y - projy)**2

    j = int(np.argmin(d2))
    d = float(np.sqrt(d2[j]))
    s = float(rc[j] + t[j]*np.sqrt(seg2[j]))
    return d, s


def label_points_with_dir(points_df, geoms, linea,
                            dir_ida="IDA", dir_vta="VUELTA",
                            switch_penalty_m=30.0, eps_s=5.0):
    """
    points_df: DataFrame con columnas LAT, LON (y opcionalmente el orden original)
    geoms: tus rutas coarse construidas con estaciones (ya existentes)
    Retorna: points_df con columnas: DIR, s_in_dir, d_to_dir, order_in_dir
    """
    assert (linea, dir_ida) in geoms and (linea, dir_vta) in geoms, "Falta geometría coarse de IDA/VUELTA"
    rI = geoms[(linea, dir_ida)]
    rV = geoms[(linea, dir_vta)]

    latv = points_df["lat"].astype(float).to_numpy()
    lonv = points_df["lon"].astype(float).to_numpy()

    dI, sI, dV, sV = [], [], [], []
    for la, lo in zip(latv, lonv):
        di, si = point_d_and_s_on_route(rI, la, lo)
        dv, sv = point_d_and_s_on_route(rV, la, lo)
        dI.append(di); sI.append(si); dV.append(dv); sV.append(sv)

    # Decisión secuencial con histéresis
    DIR = []
    s_used = []
    curr = None
    prev_s = None
    for i in range(len(latv)):
        # costos de quedarse vs cambiar
        if curr is None:
            curr = dir_ida if dI[i] <= dV[i] else dir_vta
            prev_s = sI[i] if curr == dir_ida else sV[i]
        else:
            s_curr = sI[i] if curr == dir_ida else sV[i]
            d_curr = dI[i] if curr == dir_ida else dV[i]
            cost_stay = d_curr
            if s_curr < prev_s - eps_s:
                # penaliza retroceso fuerte sobre la misma dir
                cost_stay += (prev_s - s_curr)

            s_other = sV[i] if curr == dir_ida else sI[i]
            d_other = dV[i] if curr == dir_ida else dI[i]
            cost_switch = d_other + switch_penalty_m

            if cost_switch < cost_stay:
                curr = dir_vta if curr == dir_ida else dir_ida
                prev_s = s_other
            else:
                prev_s = s_curr

        DIR.append(curr)
        s_used.append(prev_s)

    out = points_df.copy()
    out["DIR"] = DIR
    out["s_in_dir"] = s_used
    out["d_to_dir"] = [dI[i] if DIR[i]==dir_ida else dV[i] for i in range(len(DIR))]

    # Orden por s dentro de cada dir
    out["order_in_dir"] = -1
    for d in [dir_ida, dir_vta]:
        m = out["DIR"] == d
        if m.any():
            idx_sorted = out.loc[m, "s_in_dir"].sort_values().index
            out.loc[idx_sorted, "order_in_dir"] = range(len(idx_sorted))

    return out.sort_values(["DIR","order_in_dir"]).reset_index(drop=True)

def prepare_route_geoms(stations_ord):
    geoms = {}
    for (linea, d), g in stations_ord.sort_values("ORDEN").groupby(["LINEA","DIR"], sort=False):
        latv = g["lat"].astype(float).to_numpy()
        lonv = g["lon"].astype(float).to_numpy()
        if len(latv) < 2: 
            continue
        lat0, lon0 = float(latv.mean()), float(lonv.mean())
        rx, ry = ll_to_xy_m(latv, lonv, lat0, lon0)
        route_cum = cumulative_distances(rx, ry)
        geoms[(linea, d)] = dict(rx=rx, ry=ry, route_cum=route_cum, lat0=lat0, lon0=lon0,
                                 length_m=float(route_cum[-1]),
                                 is_circular=(str(d).upper()=="CIRCULAR"))
    return geoms

# Prueba
COMPLETE_LINE_CSV = "./kmz/Linea_12_completa.csv"
STATIONS_XLS= "D:\\2025\\UVG\\Tesis\\repos\\backend\\data\\Estaciones_ordenadas_with_pos.xlsx"
stations_ord = pd.read_excel(STATIONS_XLS)
stations_ord[["lat", "lon"]] = stations_ord["POSICIÓN"].apply(parse_pos)
stations_ord = stations_ord.sort_values("ORDEN")
geoms = prepare_route_geoms(stations_ord)

points_df = pd.read_csv(COMPLETE_LINE_CSV)
labeled = label_points_with_dir(points_df, geoms, "Linea 12", dir_ida="IDA", dir_vta="VUELTA")
labeled.to_csv("./complete_lines/Linea_12_completa_labeled.csv", index=False)

# Graficar con folium
import folium
from folium import PolyLine
m = folium.Map(location=[14.634915, -90.522713], zoom_start=13, tiles="OpenStreetMap")
colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "white", "pink", "lightblue", "lightgreen", "gray", "black", "lightgray"]
for i, (name, g) in enumerate(labeled.groupby("DIR")):
    color = colors[i % len(colors)]
    coords = g[["lat","lon"]].values.tolist()
    PolyLine(coords, color=color, weight=5, opacity=0.8, tooltip=name).add_to(m)
m.save("./complete_lines/Linea_12_completa_labeled_preview.html")