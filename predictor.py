"""
Mirrors notebook preprocessing exactly.
Models saved by notebook_cells.md are loaded once at startup.

FIXES APPLIED (matching notebook):
  1. Feature_Count is clipped to max 20 before inference
  2. FEATURES list drops 'Year' (redundant with Car_Age) and
     'Assembly' (redundant with Is_Imported)
  3. Z-score outlier removal for Price_PKR was removed from the
     notebook pipeline, so no equivalent filter is needed here
"""
import os, math
import numpy  as np
import joblib
from config  import MODELS_DIR
from schemas import PredictRequest, PredictResponse, FeatureImpact

# ── Notebook constants ────────────────────────────────────────────────────────
CURRENT_YEAR = 2026

# FIX 2: 'Year' and 'Assembly' removed — they were redundant with Car_Age and
# Is_Imported respectively, giving those engineered features 0% importance.
FEATURES = [
    'Fuel', 'Transmission', 'Province',
    'Body Type', 'Mileage_km', 'Engine_cc', 'Car_Age',
    'Mileage_per_Year_sqrt', 'Feature_Count',
    'Brand_Prestige', 'Engine_Cat', 'Is_Imported'
]

TIER_LABELS = {0: 'Budget', 1: 'Luxury', 2: 'Mid'}
PRESTIGE = {
    'Bentley':5,'Lamborghini':5,'Porsche':5,'BMW':5,'Mercedes':5,
    'Audi':5,'Lexus':5,'Tesla':5,'Genesis':5,'Jaguar':5,
    'Land Rover':5,'Range Rover':5,'Cadillac':5,'Volvo':5,'Infiniti':5,'GMC':5,'Hummer':5,
    'Toyota':4,'Honda':4,'Volkswagen':4,'MINI':4,'Jeep':4,'Peugeot':4,'Mazda':4,'Subaru':4,'Ford':4,
    'KIA':3,'Hyundai':3,'MG':3,'Haval':3,'Chery':3,'Nissan':3,'Mitsubishi':3,'Isuzu':3,
    'Chevrolet':3,'BYD':3,'Deepal':3,'ORA':3,'Tank':3,'Jetour':3,'JAC':3,'Dongfeng':3,
    'BAIC':3,'Seres':3,'SsangYong':3,'DFSK':3,'Sokon':3,'JMC':3,'Dodge':3,
    'Suzuki':2,'Daihatsu':2,'Proton':2,'FAW':2,'Prince':2,'United':2,'Daewoo':2,'Datsun':2,
    'Adam':2,'Master':2,'Honri':2,'Daehan':2,'Power':2,'JW':2,
    'Fiat':1,'Sogo':1,'Roma':1,'Mushtaq':1,'GUGO':1,'Opel':1,'Rinco':1,'BAW':1,'Kaiser':1,
}

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_scaler = _le_dict = _reg_model = _clf_model = _meta = None
_reg_importance: list[FeatureImpact] = []

def _path(name): return os.path.join(MODELS_DIR, name)

def models_ready() -> bool:
    return all(os.path.exists(_path(f)) for f in [
        "scaler.pkl", "le_dict.pkl", "best_reg_model.pkl", "best_clf_model.pkl", "meta.pkl"
    ])

def load_models():
    global _scaler, _le_dict, _reg_model, _clf_model, _meta, _reg_importance
    _scaler    = joblib.load(_path("scaler.pkl"))
    _le_dict   = joblib.load(_path("le_dict.pkl"))
    _reg_model = joblib.load(_path("best_reg_model.pkl"))
    _clf_model = joblib.load(_path("best_clf_model.pkl"))
    _meta      = joblib.load(_path("meta.pkl"))

    # Try direct feature_importances_ (tree-based models)
    try:
        imps = _reg_model.feature_importances_
        pairs = sorted(zip(FEATURES, imps), key=lambda x: x[1], reverse=True)
        _reg_importance = [FeatureImpact(feature=f, importance=round(float(i), 4)) for f, i in pairs[:6]]

    # Stacking / linear models — extract from the best base estimator instead
    except AttributeError:
        base = None

        # Stacking Regressor — try final_estimator first, then best base
        if hasattr(_reg_model, 'final_estimator_'):
            fe = _reg_model.final_estimator_
            if hasattr(fe, 'feature_importances_'):
                base = fe
        if base is None and hasattr(_reg_model, 'estimators_'):
            for est in _reg_model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    base = est
                    break

        if base is not None:
            imps = base.feature_importances_
            pairs = sorted(zip(FEATURES, imps), key=lambda x: x[1], reverse=True)
            _reg_importance = [FeatureImpact(feature=f, importance=round(float(i), 4)) for f, i in pairs[:6]]
        else:
            # Linear model — use absolute coefficients as proxy
            try:
                coeffs = np.abs(_reg_model.coef_)
                total  = coeffs.sum() or 1
                pairs  = sorted(zip(FEATURES, coeffs / total), key=lambda x: x[1], reverse=True)
                _reg_importance = [FeatureImpact(feature=f, importance=round(float(i), 4)) for f, i in pairs[:6]]
            except AttributeError:
                _reg_importance = [FeatureImpact(feature=f, importance=0.0) for f in FEATURES[:6]]

def get_feature_importance():
    if not models_ready(): return []
    if _scaler is None: load_models()
    return _reg_importance

# ── Feature engineering (mirrors notebook) ───────────────────────────────────
def _engine_cat(cc: int) -> str:
    if cc <= 800:  return 'Mini'
    if cc <= 1300: return 'Small'
    if cc <= 1800: return 'Medium'
    if cc <= 2500: return 'Large'
    return 'XLarge'

def _clean_province(x: str) -> str:
    x = x.lower()
    if 'punjab'    in x:                return 'Punjab'
    if 'sindh'     in x:                return 'Sindh'
    if 'khyber'    in x or 'kpk' in x: return 'KPK'
    if 'baloch'    in x:                return 'Balochistan'
    if 'gilgit'    in x:                return 'Gilgit Baltistan'
    if 'kashmir'   in x or 'ajk' in x: return 'AJK'
    if 'islamabad' in x:                return 'Islamabad'
    return 'Punjab'

def _le_encode(col: str, val: str) -> int:
    le = _le_dict[col]
    if val in le.classes_:
        return int(le.transform([val])[0])
    return int(le.transform([le.classes_[0]])[0])   # fallback

def _build_vector(req: PredictRequest) -> np.ndarray:
    car_age  = max(CURRENT_YEAR - req.year, 0)
    mpy_sqrt = math.sqrt(req.mileage_km / max(car_age, 1))

    # FIX 1: Clip Feature_Count to <=20 to match notebook preprocessing.
    # A raw value of 0 previously tanked price estimates because Feature_Count
    # carried ~47% regression importance. Clipping stabilises inference.
    feature_count_clipped = min(req.feature_count, 20)

    # FIX 2: 'Year' and 'Assembly' are no longer in FEATURES.
    # Car_Age carries the age signal; Is_Imported carries the assembly signal.
    return np.array([[
        _le_encode('Fuel',        req.fuel),
        _le_encode('Transmission', req.transmission),
        _le_encode('Province',    _clean_province(req.province)),
        _le_encode('Body Type',   req.body_type),
        req.mileage_km,
        req.engine_cc,
        car_age,
        mpy_sqrt,
        feature_count_clipped,
        PRESTIGE.get(req.make, 2),
        _le_encode('Engine_Cat',  _engine_cat(req.engine_cc)),
        1 if req.assembly.lower() == 'imported' else 0,
    ]], dtype=float)

# ── Main predict function ─────────────────────────────────────────────────────
def predict_car(req: PredictRequest) -> PredictResponse:
    if not models_ready():
        raise RuntimeError("Models not found. Run the notebook saving cells first.")
    if _scaler is None:
        load_models()

    X_raw = _build_vector(req)

    # Regression
    X_r   = _scaler.transform(X_raw) if _meta['reg_scaled'] else X_raw
    log_p = float(_reg_model.predict(X_r)[0])
    price_pkr  = int(np.expm1(log_p))
    price_lacs = round(price_pkr / 1e5, 2)

    # Classification
    X_c      = _scaler.transform(X_raw) if _meta['clf_scaled'] else X_raw
    tier_enc = int(_clf_model.predict(X_c)[0])
    tier_str = TIER_LABELS.get(tier_enc, 'Unknown')

    # Negotiation range (+-8%)
    neg_low  = round(price_lacs * 0.92, 2)
    neg_high = round(price_lacs * 1.08, 2)

    # Valuation tag
    val_tag = None
    if req.listed_price_lacs is not None:
        ratio = req.listed_price_lacs / price_lacs if price_lacs > 0 else 1
        if   ratio < 0.90: val_tag = "Underpriced"
        elif ratio > 1.10: val_tag = "Overpriced"
        else:              val_tag = "Fair"

    r2   = _meta.get('best_r2', 0.85)
    conf = round(max(0.0, min(1.0, r2)) * 100, 1)

    return PredictResponse(
        predicted_price_lacs  = price_lacs,
        predicted_price_pkr   = price_pkr,
        negotiation_low_lacs  = neg_low,
        negotiation_high_lacs = neg_high,
        price_tier            = tier_str,
        valuation_tag         = val_tag,
        confidence_pct        = conf,
        top_features          = _reg_importance[:5],
        model_used_reg        = _meta.get('reg_name', '—'),
        model_used_clf        = _meta.get('clf_name', '—'),
    )
