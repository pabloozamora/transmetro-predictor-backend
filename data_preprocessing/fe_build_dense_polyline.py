'''Script para construir una polilínea densa de una línea de transporte público
a partir del archivo CSV construido con el KML de la línea indicada, y guardar los puntos en un CSV junto con sus estaciones.'''

import pandas as pd
import folium

LINE='Linea_18-A'

line_csv=f'./kmz/{LINE}_completa.csv'
line_df = pd.read_csv(line_csv)

# Rellenar valores faltantes en la columna 'station' con nulos
line_df['station'] = line_df['station'].fillna('')

# Rellenar valores faltantes de la columna 'dir' con el valor anterior
line_df['dir'] = line_df['dir'].fillna(method='ffill')

# Guardar el DataFrame actualizado en un nuevo archivo CSV
line_df.to_csv(f"./complete_lines/{LINE}_with_direction.csv", index=False)

# Crear polilínea con folium, marcando los puntos que tienen estación
m = folium.Map(location=[line_df['lat'].mean(), line_df['lon'].mean()], zoom_start=13)

for i, row in line_df.iterrows():
    if pd.notna(row['station']):  # add marker with station name if 'station' is not empty
        folium.Marker(location=[row['lat'], row['lon']], popup=row['station']).add_to(m)
    else: # Add marker with index number
        folium.CircleMarker(location=[row['lat'], row['lon']], radius=3, color='red').add_to(m)
     # Draw line to next point if not the last point
    if i < len(line_df) - 1:
        next_row = line_df.iloc[i + 1]
        folium.PolyLine(locations=[[row['lat'], row['lon']], [next_row['lat'], next_row['lon']]], color='blue').add_to(m)
m.save(f"./complete_lines/{LINE}_with_stations.html")