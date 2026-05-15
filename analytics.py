"""
Loads PW_dataset.csv (saved by notebook cell 80) and computes
all EDA statistics needed by the dashboard.  If the CSV is absent,
every function returns None and the .NET app shows a placeholder.
"""
import os, math
import pandas as pd
import numpy  as np
from config import DATA_PATH

_df: pd.DataFrame | None = None

# ── Init ──────────────────────────────────────────────────────────────────────
def load_dataset() -> bool:
    global _df
    if os.path.exists(DATA_PATH):
        _df = pd.read_csv(DATA_PATH)
        return True
    return False

def dataset_loaded() -> bool:
    return _df is not None

# ── Dataset summary ───────────────────────────────────────────────────────────
def get_dataset_summary():
    if _df is None: return None
    return {
        "total_rows"        : int(len(_df)),
        "total_features"    : 14,
        "price_mean_lacs"   : round(float(_df["Price_PKR"].mean()) / 1e5, 2),
        "price_median_lacs" : round(float(_df["Price_PKR"].median()) / 1e5, 2),
        "price_min_lacs"    : round(float(_df["Price_PKR"].min()) / 1e5, 2),
        "price_max_lacs"    : round(float(_df["Price_PKR"].max()) / 1e5, 2),
        "avg_mileage_km"    : int(_df["Mileage_km"].mean()),
        "avg_car_age"       : round(float(_df["Car_Age"].mean()), 1),
        "year_min"          : int(_df["Year"].min()),
        "year_max"          : int(_df["Year"].max()),
        "unique_makes"      : int(_df["Make"].nunique()) if "Make" in _df.columns else 0,
    }

# ── Price distribution (histogram) ───────────────────────────────────────────
def get_price_distribution():
    if _df is None: return None
    prices = _df["Price_PKR"].dropna() / 1e5
    hist, edges = np.histogram(prices, bins=25, range=(0, prices.quantile(0.98)))
    return {
        "labels": [f"{e:.0f}L" for e in edges[:-1]],
        "values": [int(v) for v in hist],
    }

# ── Price tier distribution ───────────────────────────────────────────────────
def get_tier_distribution():
    if _df is None: return None
    # Price_Tier is label-encoded in dataset: 0=Budget,1=Luxury,2=Mid
    tier_map = {0:"Budget", 1:"Luxury", 2:"Mid"}
    if "Price_Tier" in _df.columns:
        vc = _df["Price_Tier"].value_counts().to_dict()
        return {tier_map.get(int(k), str(k)): int(v) for k,v in vc.items()}
    return None

# ── Top makes by average price ────────────────────────────────────────────────
def get_top_makes(n=12):
    if _df is None or "Make" not in _df.columns: return None
    top = (_df.groupby("Make")["Price_PKR"]
              .mean()
              .sort_values(ascending=False)
              .head(n) / 1e5)
    return {"labels": list(top.index), "values": [round(float(v),2) for v in top]}

# ── Correlation matrix (numeric features only) ────────────────────────────────
def get_correlation():
    if _df is None: return None
    num_cols = ["Price_PKR","Car_Age","Mileage_km","Engine_cc",
                "Feature_Count","Brand_Prestige","Is_Imported"]
    avail = [c for c in num_cols if c in _df.columns]
    corr  = _df[avail].corr().round(3)
    return {
        "labels": avail,
        "matrix": corr.values.tolist(),
    }

# ── Price vs Car Age scatter (sampled) ───────────────────────────────────────
def get_price_vs_age(sample=600):
    if _df is None: return None
    s = _df[["Car_Age","Price_PKR","Brand_Prestige"]].dropna().sample(
        min(sample, len(_df)), random_state=42)
    return {
        "ages"     : s["Car_Age"].astype(int).tolist(),
        "prices"   : (s["Price_PKR"]/1e5).round(2).tolist(),
        "prestige" : s["Brand_Prestige"].astype(int).tolist(),
    }

# ── Province avg price ────────────────────────────────────────────────────────
def get_province_prices():
    if _df is None or "Province" not in _df.columns: return None
    # Province is label-encoded; decode if possible
    avg = (_df.groupby("Province")["Price_PKR"]
              .mean().sort_values(ascending=False) / 1e5)
    return {"labels": [str(k) for k in avg.index],
            "values": [round(float(v),2) for v in avg]}
