from pydantic import BaseModel, Field
from typing import List, Optional

# === Esquemas de entrada/salida ===
        
class PredictRequest(BaseModel):
    target_line: str = Field(..., description="Línea objetivo para la predicción de ETA")
    target_station: str = Field(..., description="Estación objetivo para la predicción de ETA")
    target_direction: str = Field(..., description="Dirección objetivo para la predicción de ETA")
    
class PredictResponse(BaseModel):
    closest_unit: Optional[dict] = Field(
        None,
        description="Información del vehículo más cercano a la estación objetivo",
    )
    prediction: float = Field(..., description="Predicción de ETA en segundos")
    model_version: str
    n_models: int
    
class TripRequest(BaseModel):
    target_line: str = Field(..., description="Línea objetivo para la predicción de duración del viaje")
    origin_station: str = Field(..., description="Estación de origen del viaje")
    origin_direction: str = Field(..., description="Dirección de la estación de origen")
    dest_station: str = Field(..., description="Estación de destino del viaje")
    dest_direction: str = Field(..., description="Dirección objetivo para la predicción de duración del viaje")

class TripResponse(BaseModel):
    closest_unit: Optional[dict] = Field(
        None,
        description="Información del vehículo más cercano a la estación de origen",
    )
    prediction: float = Field(..., description="Predicción de duración del viaje en segundos")
    model_version: str
    n_models: int
    
class StationEntry(BaseModel):
    id: int
    name: str
    line: str
    direction: str
    lat: float
    lon: float