from fastapi import FastAPI, HTTPException
from schemes import PredictRequest, PredictResponse
from models import load_model
from predictions import get_closest_ETA_for_station

from typing import List, Optional
import pandas as pd
import numpy as np

DEMO = True # Utilizar datos demo

app = FastAPI(title="Transmetro ETA API", version="1.0.0")


# === Utilidades ===

@app.on_event("startup")
def _startup():
    app.state.boosters = load_model()
    app.state.model_version = "2025-10-25"
    
@app.post("/ETA", response_model=PredictResponse)
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "API de TrackMetro."}
