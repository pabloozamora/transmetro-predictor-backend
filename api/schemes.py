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