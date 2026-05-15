from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schemas   import (RegressionResultIn, ClassificationResultIn,
                        PredictRequest)
from database  import (db_insert_regression, db_get_all_regression, db_delete_regression,
                        db_insert_classification, db_get_all_classification, db_delete_classification)
from analytics import load_dataset, dataset_loaded, get_dataset_summary, get_price_distribution, \
                       get_tier_distribution, get_top_makes, get_correlation, \
                       get_price_vs_age, get_province_prices
from predictor import models_ready, load_models, predict_car, get_feature_importance

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dataset()
    if models_ready():
        load_models()
    yield

app = FastAPI(title="PakWheels ML API", version="2.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"         : "ok",
        "dataset_loaded" : dataset_loaded(),
        "models_ready"   : models_ready(),
    }

# ── Analytics ─────────────────────────────────────────────────────────────────
@app.get("/api/analytics/summary")
def analytics_summary():
    return {
        "dataset"          : get_dataset_summary(),
        "regression_count" : len(db_get_all_regression()),
        "clf_count"        : len(db_get_all_classification()),
        "models_ready"     : models_ready(),
    }

@app.get("/api/analytics/eda/price-distribution")
def eda_price_dist():
    data = get_price_distribution()
    if data is None:
        raise HTTPException(404, "Dataset not loaded. Run notebook first.")
    return data

@app.get("/api/analytics/eda/tier-distribution")
def eda_tier_dist():
    data = get_tier_distribution()
    if data is None:
        raise HTTPException(404, "Dataset not loaded.")
    return data

@app.get("/api/analytics/eda/top-makes")
def eda_top_makes():
    data = get_top_makes()
    if data is None:
        raise HTTPException(404, "Dataset not loaded.")
    return data

@app.get("/api/analytics/eda/correlation")
def eda_correlation():
    data = get_correlation()
    if data is None:
        raise HTTPException(404, "Dataset not loaded.")
    return data

@app.get("/api/analytics/eda/price-vs-age")
def eda_price_vs_age():
    data = get_price_vs_age()
    if data is None:
        raise HTTPException(404, "Dataset not loaded.")
    return data

@app.get("/api/analytics/eda/province-prices")
def eda_province():
    data = get_province_prices()
    if data is None:
        raise HTTPException(404, "Dataset not loaded.")
    return data

@app.get("/api/analytics/feature-importance")
def feat_importance():
    return get_feature_importance()

# ── Model results (read) ──────────────────────────────────────────────────────
@app.get("/api/results/regression")
def get_regression():
    return db_get_all_regression()

@app.get("/api/results/classification")
def get_classification():
    return db_get_all_classification()

# ── Model results (write — called by notebook) ────────────────────────────────
@app.post("/api/results/regression", status_code=201)
def post_regression(r: RegressionResultIn):
    db_insert_regression(r)
    return {"status": "saved", "model": r.model_name}

@app.post("/api/results/classification", status_code=201)
def post_classification(r: ClassificationResultIn):
    db_insert_classification(r)
    return {"status": "saved", "model": r.model_name}

@app.delete("/api/results/regression/{id}")
def del_regression(id: int):
    db_delete_regression(id)
    return {"deleted": id}

@app.delete("/api/results/classification/{id}")
def del_classification(id: int):
    db_delete_classification(id)
    return {"deleted": id}

# ── Prediction ────────────────────────────────────────────────────────────────
@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        return predict_car(req)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
