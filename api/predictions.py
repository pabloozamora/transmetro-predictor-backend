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
    
# Estadísticas predefinidas de ETA entre estaciones:
ETA_STATS_PATH = Path("D:\\2025\\UVG\\Tesis\\repos\\backend\\models\\eta_tramos_robusto.csv")

# Cargar estadísticas de ETA entre estaciones
ETA_STATS_DF = pd.read_csv(ETA_STATS_PATH)

def get_station_sequence(linea: str, dir_: str) -> list[str]:
    """Regresa la secuencia canónica de estaciones para (LINEA, DIR)."""
    return STATION_ORDER.get(str(linea), {}).get(str(dir_), [])

def find_stations_between(
    candidates: list[str],
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
    seq = get_station_sequence(target_line, target_dir)
    if not seq or target_station not in seq:
        return None

    is_bidirectional = target_dir in ("IDA", "VUELTA")
    opp_dir = "VUELTA" if target_dir == "IDA" else "IDA"
    opp_seq = get_station_sequence(target_line, opp_dir) if is_bidirectional else []

    idx_t = seq.index(target_station)
    results: list[list[str]] = []

    for st in candidates:
        # Igual a la target ⇒ nada "entre"
        if st == target_station:
            results.append([st])
            continue

        if st in seq:
            idx_c = seq.index(st)
            if idx_c < idx_t:
                # Misma dirección, candidata antes de target
                results.append([st] + seq[idx_c + 1 : idx_t] + [target_station])
            else:
                # Misma dirección, candidata después de target ⇒ wrap
                if is_bidirectional and opp_seq:
                    # Avanza en el sentido objetivo hasta su final, luego todo el opuesto, luego entra al sentido objetivo desde el inicio hasta target
                    segment = seq[idx_c + 1 :] + opp_seq + seq[:idx_t]
                else:
                    # Circular (o sin opp): wrap interno
                    segment = seq[idx_c + 1 :] + seq[:idx_t]
                results.append([st] + segment + [target_station])

        elif is_bidirectional and opp_seq and st in opp_seq:
            # Candidata en el sentido opuesto
            idx_c_opp = opp_seq.index(st)
            # Avanza en el opuesto hasta su final, luego entra al sentido objetivo desde el inicio hasta target
            segment = opp_seq[idx_c_opp + 1 :] + seq[:idx_t]
            results.append([st] + segment + [target_station])
        # Si no está en ningún sentido, se ignora la candidata

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
            print('ETA from', origin_station, 'to', dest_station, ':', eta_value)
            eta_total += eta_value
            
    return eta_total if eta_total > 0 else None

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
    
    if dir_ in ["IDA", "VUELTA"]:
        # Candidatos de la dirección opuesta
        latest_data_by_line_and_opposite_dir = get_latest_data_by_line_and_dir(line, "VUELTA" if dir_ == "IDA" else "IDA")
        
        opposite_dir = "VUELTA" if dir_ == "IDA" else "IDA"
        opposite_dir_candidate_stations = [
            data["proxima_est_teorica"]
            for (l, d, _), data in latest_data_by_line_and_opposite_dir.items()
            if l == line and d == opposite_dir
        ]
        
    same_dir_valid_candidates = find_stations_between(
        same_dir_candidate_stations, station, line, dir_
    )
    
    opposite_dir_valid_candidates = []
    if dir_ in ["IDA", "VUELTA"]:
        opposite_dir_valid_candidates = find_stations_between(
            opposite_dir_candidate_stations, station, line, dir_
        )
        
    all_valid_candidates = []
    if same_dir_valid_candidates:
        all_valid_candidates.extend(same_dir_valid_candidates)
        
    if opposite_dir_valid_candidates:
        all_valid_candidates.extend(opposite_dir_valid_candidates)
        
    # Si no hay candidatos válidos, regresar None
    if not all_valid_candidates:
        return None
        
    # Calcular ETA acumulado para cada conjunto de estaciones entre y tomar el mínimo
    best_candidate = None
    
    for valid_candidate in all_valid_candidates:
        
        candidate_ETA = 0.0
        
        latest_station = valid_candidate[0]

        # Obtener la información de la estación más cercana (latest_station) de la ruta candidata
        if latest_station in same_dir_candidate_stations:
            candidate_direction = dir_
            latest_station_key = (line, dir_, latest_station)
            latest_station_info = latest_data_by_line_and_dir.get(latest_station_key)
        else:
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
            if best_candidate is None or eta_value < best_candidate["cumulative_ETA"]:
                # Obtener placa, dir, linea, lat y lon del latest_station
                best_candidate = {
                    "unit": latest_station_info.get("Placa"),
                    "line": line,
                    "direction": candidate_direction,
                    "next_station": latest_station,
                    "cumulative_ETA": candidate_ETA,
                    "latitude": latest_station_info.get("Latitud"),
                    "longitude": latest_station_info.get("Longitud"),
                }
                    
    if best_candidate is not None:
        result = {
            "prediction": best_candidate["cumulative_ETA"],
            "closest_unit": best_candidate
        }
        
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
    
    # Luego, encontrar las estaciones entre el origen y destino
    stations_between = find_stations_between(
        candidates=[origin_station],
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

# === Ejemplo de uso ===
if __name__ == "__main__":
    line = "Linea_12"
    dir_ = "IDA"
    origin_station = "BOLÍVAR DIRECCIÓN CENTRO"
    origin_direction = "IDA"
    dest_station = "PLAZA BARRIOS"
    dest_direction = "VUELTA"

    eta = get_trip_duration_between_stations(line, origin_station, origin_direction, dest_station, dest_direction)
    if eta is not None:
        print(f"El ETA más cercano para la estación {dest_station} en la línea {line} ({origin_direction}) es: {eta} minutos.")
    else:
        print(f"No se pudo determinar el ETA para la estación {dest_station} en la línea {line} ({origin_direction}).")