import json
from pathlib import Path
from typing import Optional
import pandas as pd
from current_data import get_latest_data_by_line_and_dir
from models import predict_boosters, load_model

# Orden canónico de estaciones por línea y dirección
ORDER_PATH = Path("D:\\2025\\UVG\\Tesis\\repos\\backend\\api\\db\\station_order_by_line_dir.json")

# Cargar orden canónico de estaciones
with open(ORDER_PATH, "r", encoding="utf-8") as f:
    STATION_ORDER = json.load(f)
    
# Path de estadísticas históricas de ETA entre estaciones:
ETA_STATS_PATH = Path("D:\\2025\\UVG\\Tesis\\repos\\backend\\models\\eta_tramos_robusto.csv")

# Cargar estadísticas históricas de ETA entre estaciones
ETA_STATS_DF = pd.read_csv(ETA_STATS_PATH)

def get_station_sequence(linea: str, dir_: str) -> list[str]:
    """Regresa la secuencia canónica de estaciones para (LINEA, DIR)."""
    return STATION_ORDER.get(str(linea), {}).get(str(dir_), [])

def find_stations_between(
    candidates: list[str],
    candidates_dir: str,
    target_station: str,
    target_line: str,
    target_dir: str
) -> list[list[str]] | None:
    """
    Regresa, para cada candidata, la lista de estaciones entre la candidata y la target,
    siguiendo el orden canónico de (linea, dir). Para wrap:
      - IDA/VUELTA: resto del sentido objetivo -> TODO el sentido opuesto -> prefijo del sentido objetivo hasta target.
      - CIRCULAR: resto desde candidata -> prefijo hasta target.
    """
    # Primero, revisar si la línea es circular o bidireccional
    is_bidirectional = candidates_dir in ("IDA", "VUELTA")
    
    # Segundo, determinar si la estación objetivo está en la misma dirección o en la opuesta
    target_in_same_dir = candidates_dir == target_dir
    
    if is_bidirectional:
        candidates_opposite_dir = "VUELTA" if candidates_dir == "IDA" else "IDA"
        candidates_opp_dir_sequence = get_station_sequence(target_line, candidates_opposite_dir) # Secuencia en la dirección opuesta a los candidatos
    
    candidates_same_dir_sequence = get_station_sequence(target_line, candidates_dir) # Secuencia en la misma dirección que los candidatos

    results: list[list[str]] = []

    for st in candidates:
        # Igual a la target y en la misma dirección, no hay nada "entre". La estación objetivo es la siguiente.
        if st == target_station and target_in_same_dir:
            results.append([st])
            continue
        
        # Si la línea es bidireccional y la estación objetivo está en la misma dirección que las candidatas
        if is_bidirectional and target_in_same_dir:
            
            idx_target = candidates_same_dir_sequence.index(target_station)
            idx_candidate = candidates_same_dir_sequence.index(st)
            
            # Candidata antes de target
            if idx_candidate < idx_target:
                # Agregar las estaciones entre candidata y target
                segment = [st] + candidates_same_dir_sequence[idx_candidate + 1: idx_target + 1]
                results.append(segment)

            # Candidata después de target
            else:
                # Avanza en la dirección actual hasta su final, luego toda la opuesta, luego entra al sentido objetivo desde el inicio hasta target
                segment = [st] + candidates_same_dir_sequence[idx_candidate + 1 :] + candidates_opp_dir_sequence + candidates_same_dir_sequence[:idx_target + 1]
                results.append(segment)

        # Si la estación objetivo está en la dirección opuesta a las candidatas
        elif is_bidirectional and not target_in_same_dir:
            # Avanza en la dirección actual hasta su final, luego toda la opuesta, luego entra al sentido objetivo desde el inicio hasta target
            idx_target = candidates_opp_dir_sequence.index(target_station)
            idx_candidate = candidates_same_dir_sequence.index(st)
            segment = [st] + candidates_same_dir_sequence[idx_candidate + 1:] + candidates_opp_dir_sequence[:idx_target + 1]
            results.append(segment)
            
        else:  # Línea circular
            idx_target = candidates_same_dir_sequence.index(target_station)
            idx_candidate = candidates_same_dir_sequence.index(st)
            
            # Candidata antes de target
            if idx_candidate < idx_target:
                segment = [st] + candidates_same_dir_sequence[idx_candidate + 1 : idx_target + 1]
                results.append(segment)
            else:
                # Avanza desde candidata hasta el final, luego desde el inicio hasta target
                segment = [st] + candidates_same_dir_sequence[idx_candidate + 1 :] + candidates_same_dir_sequence[:idx_target + 1]
                results.append(segment)

    return results or None

def cumulative_ETA(stations_between: list[str], line: str) -> str | None:
    """
    Obtiene el ETA acumulado entre estaciones consecutivas.
    """
    
    # Calcular ETA acumulado para las estaciones en medio
    if not stations_between:
        return None
    
    # Acumular los ETAs entre estaciones de por medio:
    eta_total = 0.0
    for i in range(len(stations_between) - 1):
        origin_station = stations_between[i]
        dest_station = stations_between[i + 1]
        
        # Si es la misma estación, no sumar nada
        if origin_station == dest_station:
            continue
        
        eta_row = ETA_STATS_DF[
            (ETA_STATS_DF["LINEA"] == line) &
            (ETA_STATS_DF["estacion_actual"] == origin_station) &
            (ETA_STATS_DF["siguiente_estacion"] == dest_station)
        ]
        
        if not eta_row.empty:
            eta_value = eta_row["ETA_sugerido_s"].values[0]
            print('ETA histórico desde', origin_station, 'hasta', dest_station, ':', eta_value)
            eta_total += eta_value
            
    return eta_total

def get_closest_ETA_for_station(
    line: str,
    dir_: str,
    station: str,
    boosters: list,
) -> Optional[float]:
    """
    Regresa el ETA más cercano para la estación dada, acumulando ETAs estadísticos si es necesario.
    """
    # Buscar candidatos para la estación más cercana a la objetivo
    # Candidatos de la misma línea y dirección
    latest_data_by_line_and_dir = get_latest_data_by_line_and_dir(line, dir_)

    same_dir_candidate_stations = [
        data["proxima_est_teorica"]
        for (l, d, _), data in latest_data_by_line_and_dir.items()
        if l == line and d == dir_
    ]
    
    latest_data_by_line_and_opposite_dir = {}
    if dir_ in ["IDA", "VUELTA"]:
        # Candidatos de la dirección opuesta
        latest_data_by_line_and_opposite_dir = get_latest_data_by_line_and_dir(line, "VUELTA" if dir_ == "IDA" else "IDA")
        
        opposite_dir = "VUELTA" if dir_ == "IDA" else "IDA"
        opposite_dir_candidate_stations = [
            data["proxima_est_teorica"]
            for (l, d, _), data in latest_data_by_line_and_opposite_dir.items()
            if l == line and d == opposite_dir
        ]
        
    if latest_data_by_line_and_dir:
        print('\n=== Unidades disponibles para ', line, 'dirección', dir_, '===')
        for key, value in latest_data_by_line_and_dir.items():
            print('Unidad:', value.get("Placa"), 'Último registro:', value.get("Fecha"), 'Próxima estación:', value.get("proxima_est_teorica"))
            
    if latest_data_by_line_and_opposite_dir:
        print('\n=== Unidades disponibles para', line, 'dirección opuesta ===')
        for key, value in latest_data_by_line_and_opposite_dir.items():
            print('Unidad:', value.get("Placa"), 'Último registro:', value.get("Fecha"), 'Próxima estación:', value.get("proxima_est_teorica"))
        
    same_dir_valid_candidates = []
    same_dir_valid_candidates = find_stations_between(
        candidates=same_dir_candidate_stations,
        candidates_dir=dir_,
        target_station=station,
        target_line=line,
        target_dir=dir_
    )

    opposite_dir_valid_candidates = []
    if dir_ in ["IDA", "VUELTA"]:
        opposite_dir = "VUELTA" if dir_ == "IDA" else "IDA"
        opposite_dir_valid_candidates = find_stations_between(
            candidates=opposite_dir_candidate_stations,
            candidates_dir=opposite_dir,
            target_station=station,
            target_line=line,
            target_dir=dir_
        )
        
        
    # Si no hay candidatos válidos, regresar None
    if not (same_dir_valid_candidates or opposite_dir_valid_candidates):
        return None
        
    # Calcular ETA acumulado para cada conjunto de estaciones entre y tomar el mínimo
    best_candidate_same_dir = None
    best_candidate_opp_dir = None
    
    # Primero, candidatos en la misma dirección
    if same_dir_valid_candidates:
        for valid_candidate in same_dir_valid_candidates:
            
            candidate_ETA = 0.0
            
            latest_station = valid_candidate[0]

            candidate_direction = dir_
            latest_station_key = (line, dir_, latest_station)
            latest_station_info = latest_data_by_line_and_dir.get(latest_station_key)
                
            # Predecir el ETA de la primera estación candidata
            latest_ETA_value = predict_boosters(
                boosters,
                latest_station_info
            )
            
            candidate_ETA += latest_ETA_value[0]
            
            # Acumular ETAs estadísticos desde la estación candidata hasta la estación objetivo
            
            eta_value = cumulative_ETA(valid_candidate, line)
            if eta_value is not None:
                candidate_ETA += eta_value
                if best_candidate_same_dir is None or eta_value < best_candidate_same_dir["cumulative_ETA"]:
                    # Obtener placa, dir, linea, lat y lon del latest_station
                    best_candidate_same_dir = {
                        "unit": latest_station_info.get("Placa"),
                        "latest_timestamp": latest_station_info.get("Fecha"),
                        "line": line,
                        "direction": candidate_direction,
                        "next_station": latest_station,
                        "next_station_ETA": latest_ETA_value[0],
                        "cumulative_ETA": candidate_ETA,
                        "latitude": latest_station_info.get("Latitud"),
                        "longitude": latest_station_info.get("Longitud"),
                        "stations_between": valid_candidate
                    }
                print('Para la unidad', latest_station_info.get("Placa"), 'el ETA acumulado es:', candidate_ETA)
                
    # Segundo, candidatos en la dirección opuesta
    if opposite_dir_valid_candidates:
        for valid_candidate in opposite_dir_valid_candidates:
        
            candidate_ETA = 0.0
            
            latest_station = valid_candidate[0]

            candidate_direction = "VUELTA" if dir_ == "IDA" else "IDA"
            latest_station_key = (line, candidate_direction, latest_station)
            latest_station_info = latest_data_by_line_and_opposite_dir.get(latest_station_key)
                
            # Predecir el ETA de la primera estación candidata
            latest_ETA_value = predict_boosters(
                boosters,
                latest_station_info
            )
            
            candidate_ETA += latest_ETA_value[0]
            
            # Acumular ETAs estadísticos desde la estación candidata hasta la estación objetivo
            
            eta_value = cumulative_ETA(valid_candidate, line)
            if eta_value is not None:
                candidate_ETA += eta_value
                if best_candidate_opp_dir is None or eta_value < best_candidate_opp_dir["cumulative_ETA"]:
                    # Obtener placa, dir, linea, lat y lon del latest_station
                    best_candidate_opp_dir = {
                        "unit": latest_station_info.get("Placa"),
                        "latest_timestamp": latest_station_info.get("Fecha"),
                        "line": line,
                        "direction": candidate_direction,
                        "next_station": latest_station,
                        "next_station_ETA": latest_ETA_value[0],
                        "cumulative_ETA": candidate_ETA,
                        "latitude": latest_station_info.get("Latitud"),
                        "longitude": latest_station_info.get("Longitud"),
                        "stations_between": valid_candidate
                    }
                print('Para la unidad', latest_station_info.get("Placa"), 'el ETA acumulado es:', candidate_ETA)
                    
    # Comparar ambos mejores candidatos y regresar el mejor
    if best_candidate_same_dir and best_candidate_opp_dir:
        
        if best_candidate_same_dir["cumulative_ETA"] <= best_candidate_opp_dir["cumulative_ETA"]:
            result = {
                "prediction": best_candidate_same_dir["cumulative_ETA"],
                "closest_unit": best_candidate_same_dir
            }
            print('\nEl mejor candidato es la unidad:', best_candidate_same_dir["unit"], 'con un ETA total de:', best_candidate_same_dir["cumulative_ETA"])
            return result
        
        else:
            result = {
                "prediction": best_candidate_opp_dir["cumulative_ETA"],
                "closest_unit": best_candidate_opp_dir
            }
            print('\nEl mejor candidato es la unidad:', best_candidate_opp_dir["unit"], 'con un ETA total de:', best_candidate_opp_dir["cumulative_ETA"])
            return result
        
    elif best_candidate_same_dir:
        result = {
            "prediction": best_candidate_same_dir["cumulative_ETA"],
            "closest_unit": best_candidate_same_dir
        }
        print('\nEl mejor candidato es la unidad:', best_candidate_same_dir["unit"], 'con un ETA total de:', best_candidate_same_dir["cumulative_ETA"])
        return result
    
    elif best_candidate_opp_dir:
        result = {
            "prediction": best_candidate_opp_dir["cumulative_ETA"],
            "closest_unit": best_candidate_opp_dir
        }
        print('\nEl mejor candidato es la unidad:', best_candidate_opp_dir["unit"], 'con un ETA total de:', best_candidate_opp_dir["cumulative_ETA"])
        return result
    
    return None

def get_trip_duration_between_stations(
    line: str,
    origin_station: str,
    origin_direction: str,
    dest_station: str,
    dest_direction: str
) -> Optional[float]:
    """
    Obtiene la duración estimada del viaje entre dos estaciones en una línea y dirección dadas.
    """
    
    # Primero, estimar el ETA a la estación más cercana al origen
    closest_eta = get_closest_ETA_for_station(line, origin_direction, origin_station, boosters=load_model())
    if closest_eta is None:
        return None
    
    current_ETA = closest_eta["prediction"]
    best_candidate = closest_eta["closest_unit"]
    
    print('\n === Calculando duración del viaje entre estaciones === ')
    # Luego, encontrar las estaciones entre el origen y destino
    stations_between = find_stations_between(
        candidates=[origin_station],
        candidates_dir=origin_direction,
        target_line=line,
        target_dir=dest_direction,
        target_station=dest_station,
        
    )
    
    if not stations_between:
        return None
    
    cumulative_eta = cumulative_ETA(stations_between[0], line)
    
    if cumulative_eta is None:
        return None
    
    current_ETA += cumulative_eta
    
    print('Duración estimada del viaje entre', origin_station, 'y', dest_station, ':', current_ETA)
    
    result = {
        "prediction": current_ETA,
        "closest_unit": best_candidate
    }
    
    return result