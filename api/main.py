from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from schemes import PredictRequest, PredictResponse, TripRequest, TripResponse, StationEntry
from models import load_model
from predictions import get_closest_ETA_for_station, get_trip_duration_between_stations

from typing import List, Literal, Optional
import pandas as pd
import numpy as np
import json

STATIONS_PATH = "D:/2025/UVG/Tesis/repos/backend/api/db/stations.json"

DEMO = True # Utilizar datos demo

app = FastAPI(title="Transmetro ETA API", version="1.0.0")

# === Middlewares ===

origins = [
    "http://localhost:5173",  # Vite default
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # O usa allow_origin_regex para subdominios
    allow_credentials=True,         # si usarás cookies/sesión
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Utilidades ===

@app.on_event("startup")
def _startup():
    app.state.boosters = load_model()
    app.state.model_version = "2025-10-25"
    
@app.post("/eta", response_model=PredictResponse)
def predict(req: PredictRequest):
    
    line = req.target_line
    station = req.target_station
    dir_ = req.target_direction

    if not line:
        raise HTTPException(400, "Falta 'target_line' en la solicitud")
    
    if not station:
        raise HTTPException(400, "Falta 'target_station' en la solicitud")
    
    if not dir_:
        raise HTTPException(400, "Falta 'target_direction' en la solicitud")
    
    result = get_closest_ETA_for_station(
        line=line,
        station=station,
        dir_=dir_,
        boosters=app.state.boosters,
    )
    
    if not result:
        raise HTTPException(404, "No hay unidades operativas para la línea dada.")
    
    return PredictResponse(
        prediction=result["prediction"],
        closest_unit=result["closest_unit"],
        model_version=app.state.model_version,
        n_models=len(app.state.boosters)
)
    
@app.post("/trip", response_model=TripResponse)
def predict_trip(req: TripRequest):
    line = req.target_line
    origin_station = req.origin_station
    dest_station = req.dest_station
    origin_direction = req.origin_direction
    dest_direction = req.dest_direction

    if not line:
        raise HTTPException(400, "Falta 'target_line' en la solicitud")

    if not origin_station:
        raise HTTPException(400, "Falta 'origin_station' en la solicitud")
    
    if not origin_direction:
        raise HTTPException(400, "Falta 'origin_direction' en la solicitud")

    if not dest_station:
        raise HTTPException(400, "Falta 'dest_station' en la solicitud")

    if not dest_direction:
        raise HTTPException(400, "Falta 'dest_direction' en la solicitud")

    result = get_trip_duration_between_stations(
        line=line,
        origin_station=origin_station,
        origin_direction=origin_direction,
        dest_direction=dest_direction,
        dest_station=dest_station
    )

    if result is None:
        raise HTTPException(404, "No se pudo calcular la duración del viaje.")

    return TripResponse(
        prediction=result["prediction"],
        closest_unit=result["closest_unit"],
        model_version=app.state.model_version,
        n_models=len(app.state.boosters)
    )

@lru_cache
def _load_stations_by_line() -> dict:
    with open(STATIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/stations", response_model=List[StationEntry])
def get_stations(
    line: Optional[str] = None,
    direction: Optional[Literal["IDA", "VUELTA"]] = None
):
    data = _load_stations_by_line()
    out: List[StationEntry] = []

    for entry in data:
        if line and entry["line"] != line:
            continue
        if direction and entry["direction"] != direction:
            continue
        out.append(StationEntry(
            id=entry["id"],
            name=entry["name"],
            line=entry["line"],
            direction=entry["direction"],
            lat=entry["lat"],
            lon=entry["lon"],
        ))

    return out

@app.get("/stations/line/{line}", response_model=List[StationEntry])
def get_stations_by_line(line: str) -> List[StationEntry]:
    data = _load_stations_by_line()
    out: List[StationEntry] = []

    for entry in data:
        if entry["line"] != line:
            continue
        out.append(StationEntry(
            id=entry["id"],
            name=entry["name"],
            line=entry["line"],
            direction=entry["direction"],
            lat=entry["lat"],
            lon=entry["lon"],
        ))

    return out

@app.get("/lines", response_model=List[str])
def get_lines() -> List[str]:
    data = _load_stations_by_line()
    lines = set()
    for entry in data:
        lines.add(entry["line"])
    return sorted(list(lines))

@app.get("/stations/{station_id}", response_model=StationEntry)
def get_station(station_id: int):
    data = _load_stations_by_line()
    for entry in data:
        if entry["id"] == station_id:
            return StationEntry(
                id=entry["id"],
                name=entry["name"],
                line=entry["line"],
                direction=entry["direction"],
                lat=entry["lat"],
                lon=entry["lon"],
            )
    raise HTTPException(404, "Estación no encontrada")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "API de TrackMetro."}
