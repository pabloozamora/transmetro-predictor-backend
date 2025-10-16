import pandas as pd
import folium
from datetime import datetime
import numpy as np

def create_trip_polylines_map():
    """
    Create a folium map with polylines showing GPS coordinates for each trip
    up to the start timestamp (t_start_ts) from u158_trip_routes.csv
    """
    
    print("Loading trip routes data...")
    # Load the trip routes data
    trip_routes = pd.read_csv(r'd:\2025\UVG\Tesis\repos\backend\data_with_features\u158\u158_trip_routes.csv')
    
    print(f"Found {len(trip_routes)} trips in the routes data")
    
    # Load the clean trips data in chunks to handle the large file
    print("Loading clean trips data...")
    
    # Convert t_start_ts to datetime for easier comparison
    trip_routes['t_start_ts'] = pd.to_datetime(trip_routes['t_start_ts'])
    
    # Initialize the map centered on Guatemala City (approximate location)
    center_lat = 14.6349
    center_lon = -90.5069
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Color palette for different trips
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Process each trip
    for idx, trip_row in trip_routes.iterrows():
        trip_id = trip_row['trip_id']
        t_start_ts = trip_row['t_start_ts']
        linea = trip_row['LINEA']
        direccion = trip_row['DIR']
        
        print(f"Processing trip {trip_id} ({linea} - {direccion}) - Start: {t_start_ts}")
        
        # Read the clean trips data for this specific trip
        # Using chunked reading to handle large file
        trip_points = []
        chunk_size = 10000
        
        for chunk in pd.read_csv(r'd:\2025\UVG\Tesis\repos\backend\clean_data\u158\u158_clean_trips.csv', 
                                chunksize=chunk_size):
            # Filter for this specific trip
            trip_chunk = chunk[chunk['trip_id'] == trip_id].copy()
            
            if not trip_chunk.empty:
                # Convert Fecha to datetime
                trip_chunk['Fecha'] = pd.to_datetime(trip_chunk['Fecha'])
                
                # Filter points up to t_start_ts
                filtered_chunk = trip_chunk[trip_chunk['Fecha'] <= t_start_ts]
                
                if not filtered_chunk.empty:
                    # Extract coordinates
                    coords = filtered_chunk[['Latitud', 'Longitud', 'Fecha']].values
                    trip_points.extend(coords)
        
        if trip_points:
            # Sort by timestamp to ensure correct order
            trip_points = sorted(trip_points, key=lambda x: x[2])
            
            # Extract just the lat/lon coordinates
            coordinates = [[point[0], point[1]] for point in trip_points]
            
            # Skip if we don't have enough points for a meaningful polyline
            if len(coordinates) < 2:
                print(f"  Trip {trip_id}: Not enough points ({len(coordinates)})")
                continue
            
            print(f"  Trip {trip_id}: Found {len(coordinates)} GPS points")
            
            # Choose color for this trip
            color = colors[idx % len(colors)]
            
            # Create polyline
            polyline = folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Trip {trip_id} - {linea} ({direccion})<br>Points: {len(coordinates)}<br>Start: {t_start_ts}"
            )
            
            # Add to map
            polyline.add_to(m)
            
            # Add start marker
            start_marker = folium.Marker(
                location=coordinates[0],
                popup=f"Start - Trip {trip_id}<br>{linea} ({direccion})<br>{t_start_ts}",
                icon=folium.Icon(color='green', icon='play')
            )
            start_marker.add_to(m)
            
            # Add end marker
            end_marker = folium.Marker(
                location=coordinates[-1],
                popup=f"End (t_start_ts) - Trip {trip_id}<br>{linea} ({direccion})<br>{trip_points[-1][2]}",
                icon=folium.Icon(color='red', icon='stop')
            )
            end_marker.add_to(m)
        
        else:
            print(f"  Trip {trip_id}: No GPS points found up to t_start_ts")
    
    # Add a legend
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
    
    # Save the map
    output_file = r'd:\2025\UVG\Tesis\repos\backend\u158_trip_polylines_map.html'
    m.save(output_file)
    
    print(f"\nMap saved to: {output_file}")
    print("You can open this file in a web browser to view the interactive map.")
    
    return output_file

if __name__ == "__main__":
    create_trip_polylines_map()