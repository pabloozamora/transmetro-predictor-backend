import pandas as pd
from pathlib import Path

def check_for_cejusa(unit):
    print(f"=== Comprobando CEJUSA para la unidad {unit} ===")
    CLEAN_DATA_DIR = Path("D:/2025/UVG/Tesis/repos/backend/data_with_features")
    unit_dir = CLEAN_DATA_DIR / unit
    unit_clean_data = unit_dir / f"{unit}_trips_with_next_station.csv"

    unit_df = pd.read_csv(unit_clean_data, usecols=["estacion_cercana"])
    cejusa_count = unit_df['estacion_cercana'] == 'CEJUSA ANDÉN SUR '
    total_count = len(unit_df)
    
    if cejusa_count.any():
        cejusa_occurrences = cejusa_count.sum()
        print(f"La unidad {unit} tiene {cejusa_occurrences} ocurrencias de 'CEJUSA ANDÉN SUR ' en {total_count} registros.")

if __name__ == "__main__":
    print('=== Iniciando procesamiento de líneas con Viterbi ===')
    # Encontrar todas las unidades (carpetas en clean_data)
    DATA_WITH_FEATURES_PATH = Path("D:/2025/UVG/Tesis/repos/backend/data_with_features")
    if not DATA_WITH_FEATURES_PATH.exists():
        print(f"No existe el directorio {DATA_WITH_FEATURES_PATH}. Fin.")
    units = [p.name for p in DATA_WITH_FEATURES_PATH.iterdir() if p.is_dir() and p.name != "maps"]
    if not units:
        print(f"No se encontraron unidades en {DATA_WITH_FEATURES_PATH}. Fin.")
    for unit in units:
        check_for_cejusa(unit)