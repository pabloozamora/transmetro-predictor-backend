import zipfile, io, xml.etree.ElementTree as ET, pandas as pd, numpy as np
from pathlib import Path

ROUTE_NAME = "Linea_18-B"

kmz_path = Path(f"./{ROUTE_NAME}.kmz")
assert kmz_path.exists(), f"KMZ no encontrado en {kmz_path}"

with zipfile.ZipFile(kmz_path, "r") as z:
    kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
    other = [n for n in z.namelist() if not n.lower().endswith(".kml")]
    kml_names_sorted = sorted(kml_names)
    # Leer todos los KMLs (algunos KMZ tienen varios KMLs)
    kml_blobs = [(name, z.read(name)) for name in kml_names_sorted]

ns = {"kml": "http://www.opengis.net/kml/2.2",
      "gx": "http://www.google.com/kml/ext/2.2"}

def parse_coordinates(coord_text):
    pts = []
    if not coord_text:
        return pts
    for token in coord_text.strip().replace("\n"," ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            lon = float(parts[0]); lat = float(parts[1])
            pts.append((lat, lon))
    return pts

def collect_features(kml_bytes):
    root = ET.fromstring(kml_bytes)
    feats = []
    # Recolectar NetworkLinks
    nlinks = []
    for nl in root.findall(".//kml:NetworkLink", ns):
        href = None
        link = nl.find(".//kml:Link/kml:href", ns) or nl.find(".//kml:Url/kml:href", ns)
        if link is not None and link.text:
            href = link.text.strip()
        nname = nl.find("kml:name", ns)
        nlinks.append({"name": (nname.text.strip() if nname is not None else None), "href": href})
    # Recolectar LineStrings
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        pname = name_el.text.strip() if name_el is not None and name_el.text else None
        # Standard LineString
        for ls in pm.findall(".//kml:LineString", ns):
            coords_el = ls.find("kml:coordinates", ns)
            coords = parse_coordinates(coords_el.text if coords_el is not None else "")
            feats.append({"type":"LineString", "name":pname, "npts":len(coords), "coords":coords})
        # gx:Track
        for tr in pm.findall(".//gx:Track", ns):
            coords = []
            for c in tr.findall("gx:coord", ns):
                parts = c.text.strip().split()
                if len(parts) >= 2:
                    lon = float(parts[0]); lat = float(parts[1])
                    coords.append((lat,lon))
            feats.append({"type":"gx:Track", "name":pname, "npts":len(coords), "coords":coords})
        # Polígono (outerBoundary)
        for poly in pm.findall(".//kml:Polygon", ns):
            coords_el = poly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
            coords = parse_coordinates(coords_el.text if coords_el is not None else "")
            feats.append({"type":"Polygon", "name":pname, "npts":len(coords), "coords":coords})
    return feats, nlinks

all_rows = []
all_network = []
for name, blob in kml_blobs:
    feats, nlinks = collect_features(blob)
    for f in feats:
        for order, (lat, lon) in enumerate(f["coords"]):
            all_rows.append({"kml_file": name, "type": f["type"], "name": f["name"], "order": order, "lat": lat, "lon": lon})
    for nl in nlinks:
        all_network.append({"kml_file": name, **nl})

df_points = pd.DataFrame(all_rows)
df_network = pd.DataFrame(all_network)
df_summary = (df_points.groupby(["kml_file","type","name"], as_index=False)
                        .agg(n_points=("order","max"))
                        .assign(n_points=lambda x: x["n_points"]+1))

# Guardar CSV con los puntos extraídos
df_points.to_csv(f"./{ROUTE_NAME}_completa.csv", index=False)

# Graficar las polilíneas con folium, marcando cada punto en la polilínea con su índice
import folium
from folium import PolyLine, Marker

m = folium.Map(location=[14.634915, -90.522713], zoom_start=13, tiles="OpenStreetMap")
colors = ["red", "blue", "green", "purple", "orange", "darkred", "lightred", "beige", "darkblue", "darkgreen", "cadetblue", "darkpurple", "white", "pink", "lightblue", "lightgreen", "gray", "black", "lightgray"]

for i, (name, g) in enumerate(df_points.groupby("name")):
    color = colors[i % len(colors)]
    coords = g[["lat","lon"]].values.tolist()
    PolyLine(coords, color=color, weight=5, opacity=0.8, tooltip=name).add_to(m)
    # Mark each point with its index
    for idx, (lat, lon) in enumerate(coords):
        Marker(
            location=[lat, lon],
            tooltip=f"{name} - {idx}"
        ).add_to(m)

m.save(f"./kmz_{ROUTE_NAME}_preview.html")
