from pydantic import BaseModel, Field
from typing   import Optional, List
from datetime import datetime

# ── Results (from notebook / manual entry) ────────────────────────────────────
class RegressionResultIn(BaseModel):
    model_name : str
    r2         : float
    rmse_log   : float
    mae_log    : float
    rmse_lacs  : float
    mae_lacs   : float

class ClassificationResultIn(BaseModel):
    model_name : str
    accuracy   : float
    f1_score   : float
    precision  : float
    recall     : float

class ResultOut(BaseModel):
    id         : int
    created_at : Optional[datetime]

class RegressionResultOut(RegressionResultIn, ResultOut):
    pass

class ClassificationResultOut(ClassificationResultIn, ResultOut):
    pass

# ── Prediction ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    year          : int   = Field(..., ge=2000, le=2026, example=2021)
    make          : str   = Field(..., example="Toyota")
    fuel          : str   = Field(..., example="Petrol")
    transmission  : str   = Field(..., example="Automatic")
    province      : str   = Field(..., example="Punjab")
    assembly      : str   = Field(..., example="Local")
    body_type     : str   = Field(..., example="Sedan")
    mileage_km    : float = Field(..., ge=0, example=55000)
    engine_cc     : int   = Field(..., ge=500, le=6000, example=1800)
    feature_count : int   = Field(..., ge=0, example=10)
    listed_price_lacs: Optional[float] = Field(None, example=28.5)

class FeatureImpact(BaseModel):
    feature    : str
    importance : float

class PredictResponse(BaseModel):
    predicted_price_lacs     : float
    predicted_price_pkr      : int
    negotiation_low_lacs     : float
    negotiation_high_lacs    : float
    price_tier               : str        # Budget / Mid / Luxury
    valuation_tag            : Optional[str]   # Underpriced / Fair / Overpriced
    confidence_pct           : float
    top_features             : List[FeatureImpact]
    model_used_reg           : str
    model_used_clf           : str
