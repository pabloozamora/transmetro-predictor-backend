'''Script para crear un mapa interactivo con polilíneas que representan las rutas de los viajes
hasta el timestamp de inicio (t_start_ts) de cada viaje.'''

import pandas as pd
import folium
from datetime import datetime
import numpy as np

def create_trip_polylines_map():
    """
    Crear un mapa interactivo con polilíneas que representan las rutas de los viajes
    hasta el timestamp de inicio (t_start_ts) de cada viaje.
    """
    
    print("Cargando datos de rutas de viajes...")
    # Cargar los datos de rutas de viajes
    trip_routes = pd.read_csv(r'd:\2025\UVG\Tesis\repos\backend\data_with_features\u158\u158_trip_routes.csv')
    
    print(f"Se encontraron {len(trip_routes)} viajes en los datos de rutas")
    
    # Cargar los datos de viajes limpios en chunks para manejar el archivo grande
    print("Cargando datos de viajes limpios...")
    
    # Convertir t_start_ts a datetime para facilitar la comparación
    trip_routes['t_start_ts'] = pd.to_datetime(trip_routes['t_start_ts'])
    
    # Inicializar el mapa centrado en Ciudad de Guatemala (ubicación aproximada)
    center_lat = 14.6349
    center_lon = -90.5069
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Paleta de colores para diferentes viajes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Procesar cada viaje
    for idx, trip_row in trip_routes.iterrows():
        trip_id = trip_row['trip_id']
        t_start_ts = trip_row['t_start_ts']
        linea = trip_row['LINEA']
        direccion = trip_row['DIR']
        
        print(f"Procesando viaje {trip_id} ({linea} - {direccion}) - Inicio: {t_start_ts}")
        
        # Leer los datos de viajes limpios para este viaje específico
        # Usando lectura por chunks para manejar archivos grandes
        trip_points = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(r'd:\2025\UVG\Tesis\repos\backend\clean_data\u158\u158_clean_trips.csv', 
                                chunksize=chunk_size):
            # Filtrar para este viaje específico
            trip_chunk = chunk[chunk['trip_id'] == trip_id].copy()
            
            if not trip_chunk.empty:
                # Convertir Fecha a datetime
                trip_chunk['Fecha'] = pd.to_datetime(trip_chunk['Fecha'])
                
                # Filtrar puntos hasta t_start_ts
                filtered_chunk = trip_chunk[trip_chunk['Fecha'] <= t_start_ts]
                
                if not filtered_chunk.empty:
                    # Extraer coordenadas
                    coords = filtered_chunk[['Latitud', 'Longitud', 'Fecha']].values
                    trip_points.extend(coords)
        
        if trip_points:
            # Ordenar por timestamp para asegurar el orden correcto
            trip_points = sorted(trip_points, key=lambda x: x[2])
            
            # Extraer solo las coordenadas lat/lon
            coordinates = [[point[0], point[1]] for point in trip_points]
            
            # Omitir si no tenemos suficientes puntos para una polilínea significativa
            if len(coordinates) < 2:
                print(f"  Trip {trip_id}: No hay suficientes puntos ({len(coordinates)})")
                continue
            
            print(f"  Trip {trip_id}: Se encontraron {len(coordinates)} puntos GPS")
            
            # Elegir color para este viaje
            color = colors[idx % len(colors)]
            
            # Crear polilínea
            polyline = folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Trip {trip_id} - {linea} ({direccion})<br>Points: {len(coordinates)}<br>Start: {t_start_ts}"
            )
            
            # Agregar al mapa
            polyline.add_to(m)
            
            # Agregar marcador de inicio
            start_marker = folium.Marker(
                location=coordinates[0],
                popup=f"Inicio - Viaje {trip_id}<br>{linea} ({direccion})<br>{t_start_ts}",
                icon=folium.Icon(color='green', icon='play')
            )
            start_marker.add_to(m)
            
            # Agregar marcador de fin
            end_marker = folium.Marker(
                location=coordinates[-1],
                popup=f"Fin (t_start_ts) - Viaje {trip_id}<br>{linea} ({direccion})<br>{trip_points[-1][2]}",
                icon=folium.Icon(color='red', icon='stop')
            )
            end_marker.add_to(m)
        
        else:
            print(f"  Viaje {trip_id}: No se encontraron puntos GPS hasta t_start_ts")
    
    # Agregar una leyenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px
                ">
    <p><b>Trip Visualization Legend</b></p>
    <p><i class="fa fa-play" style="color:green"></i> Trip Start</p>
    <p><i class="fa fa-stop" style="color:red"></i> Trip End (at t_start_ts)</p>
    <p>Polylines show GPS track up to t_start_ts</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar el mapa
    output_file = r'd:\2025\UVG\Tesis\repos\backend\u158_trip_polylines_map.html'
    m.save(output_file)
    
    print(f"\nMapa guardado en: {output_file}")
    print("Puedes abrir este archivo en un navegador web para ver el mapa interactivo.")
    
    return output_file

if __name__ == "__main__":
    create_trip_polylines_map()