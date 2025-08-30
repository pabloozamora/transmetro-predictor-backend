import os
import shutil
import time
from datetime import datetime, timedelta
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------- CONFIGURACIÓN GENERAL --------------------

chrome_driver_path = r"D:\chromedriver\chromedriver.exe"
download_path = r"D:\Descargas"
csv_path = './data/valid_units.csv'

general_start_date = datetime(2024, 1, 1)
abs_max_date = datetime(2025, 4, 30)

# ----------------- FUNCIONES AUXILIARES ---------------------

def wait_download():
    while True:
        time.sleep(1)
        downloading = [f for f in os.listdir(download_path) if f.endswith(".crdownload")]
        if not downloading:
            break

def move_last_file(unit_directory, destination_name):
    files = [os.path.join(download_path, f) for f in os.listdir(download_path) if f.endswith(".xlsx")]
    if not files:
        print("⚠ No se encontró archivo descargado")
        return
    most_recent_file = max(files, key=os.path.getctime)
    new_path = os.path.join(unit_directory, destination_name)
    shutil.move(most_recent_file, new_path)
    print(f"✅ Archivo movido a {new_path}")

# ------------------------------------------------------------

# Leer el archivo CSV con las unidades y fechas
df = pd.read_csv(csv_path, sep=';')

# Normalizar formato de fecha
df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

# Filtrar placas válidas (por si hay fechas NaN)
df_validas = df.dropna(subset=['Fecha'])

# Ordenar por placa
df_validas = df_validas.sort_values(by=['Placa'])

# ------------------------------------------------------------

# Crear el navegador conectado a la sesión manual
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# ------------------------------------------------------------

# Función principal que descarga una unidad completa
def download_unit(plate, fecha_inicio, fecha_fin_total):
    print(f"\n🟢 Procesando unidad u{plate} desde {fecha_inicio.date()} hasta {fecha_fin_total.date()}")

    # Crear carpeta de la unidad
    carpeta_unidad = os.path.join(download_path, f"u{plate}")
    os.makedirs(carpeta_unidad, exist_ok=True)

    dias_maximos = 30
    rango_actual = fecha_inicio

    while rango_actual <= fecha_fin_total:
        inicio_rango = rango_actual
        fin_rango = inicio_rango + timedelta(days=dias_maximos - 1)
        if fin_rango > fecha_fin_total:
            fin_rango = fecha_fin_total

        exito = descargar_rango(plate, inicio_rango, fin_rango, carpeta_unidad)

        if not exito:
            semana_inicio = inicio_rango
            while semana_inicio <= fin_rango:
                semana_fin = min(semana_inicio + timedelta(days=6), fin_rango)
                print(f"⏳ Probando semana: {semana_inicio.date()} - {semana_fin.date()}")
                exito_semana = descargar_rango(plate, semana_inicio, semana_fin, carpeta_unidad)

                if not exito_semana:
                    dia_inicio = semana_inicio
                    while dia_inicio <= semana_fin:
                        dia_fin = dia_inicio
                        print(f"⏳ Probando día: {dia_inicio.date()}")
                        descargar_rango(plate, dia_inicio, dia_fin, carpeta_unidad)
                        dia_inicio += timedelta(days=1)
                semana_inicio += timedelta(days=7)

        rango_actual = fin_rango + timedelta(days=1)

# ------------------------------------------------------------

# Función de descarga para un rango específico
def descargar_rango(plate, start_date, end_date, unit_directory):
    wait = WebDriverWait(driver, 30)

    # Entrar al iframe
    wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "productIframe")))

    wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".x-mask")))

    # Rellenar la placa
    plate_field = wait.until(EC.element_to_be_clickable((By.ID, "txtVhclPlateVehicleSimulation-inputEl")))
    plate_field.clear()
    plate_field.send_keys(plate)

    # Rellenar fechas
    start_date_field = wait.until(EC.element_to_be_clickable((By.ID, "txtTimeInSearch-inputEl")))
    start_date_field.clear()
    start_date_field.send_keys(start_date.strftime("%Y-%m-%d"))

    end_date_field = wait.until(EC.element_to_be_clickable((By.ID, "txtTimeOutSearch-inputEl")))
    end_date_field.clear()
    end_date_field.send_keys(end_date.strftime("%Y-%m-%d"))

    # Presionar el botón de descarga
    download_button = wait.until(EC.element_to_be_clickable((By.ID, "btnDwnldSimExcel")))
    download_button.click()

    original_window = driver.current_window_handle
    try:
        wait.until(lambda d: len(d.window_handles) > 1)
        new_window = [w for w in driver.window_handles if w != original_window][0]
        driver.switch_to.window(new_window)

        for i in range(15):
            time.sleep(1)
            if len(driver.window_handles) == 1:
                driver.switch_to.window(original_window)
                print(f"✅ Descarga exitosa: {start_date.date()} - {end_date.date()}")
                wait_download()
                file_name = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.xlsx"
                move_last_file(unit_directory, file_name)
                return True

        print(f"⚠️ Falló el rango: {start_date.date()} - {end_date.date()}")
        driver.close()
        driver.switch_to.window(original_window)
        return False

    except Exception as e:
        print(f"⚠️ Error en pestaña: {e}")
        return False

# -------------- LOOP PRINCIPAL DE UNIDADES ------------------

for index, row in df_validas.iterrows():
    plate = row['Placa']
    last_date = row['Fecha']
    
    unit_end_date = min(last_date, abs_max_date)
    
    download_unit(plate, general_start_date, unit_end_date)

# ------------------------------------------------------------

print("\nPROCESO COMPLETO DE TODAS LAS UNIDADES")
time.sleep(2)
driver.quit()
