# Notebook Cells to Add at the End

Copy-paste each block as a new code cell at the very end of your notebook,
after all models have been trained and `res_df` / `res_clf_df` exist.

---

## Cell A — Install & imports for saving

```python
import joblib, os, requests

MODELS_DIR = r"C:\path\to\PakWheelsAPI\models"   # ← change this path
API_URL    = "http://localhost:8000"               # FastAPI must be running
os.makedirs(MODELS_DIR, exist_ok=True))
```

---

## Cell B — Save scaler + label encoders

```python
joblib.dump(scaler,   os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le_dict,  os.path.join(MODELS_DIR, "le_dict.pkl"))
print("Scaler and label encoders saved.")
```

---

## Cell C — Pick best models and save

```python
# Map of model name → (object, uses_scaled_data)
model_map_reg = {
    'Linear Regression'   : (lr,       True),
    'Ridge Regression'    : (ridge,    True),
    'Lasso Regression'    : (lasso,    True),
    'Bayesian Regression' : (bayes,    True),
    'Polynomial Regression':(poly_reg, True),
    'KNN Regressor'       : (knn,      True),
    'Decision Tree'       : (dt,       False),
    'Random Forest'       : (rf,       False),
    'Gradient Boosting'   : (gb,       False),
    'AdaBoost'            : (ada,      False),
    'XGBoost Regressor'   : (xgb_r,   False),
    'CatBoost Regressor'  : (cat_r,   False),
    'Bagging Regressor'   : (bag_r,   False),
    'Stacking Regressor'  : (stack_r, True),
}

model_map_clf = {
    'Logistic Regression' : (log_reg, True),
    'KNN Classifier'      : (knn,     True),
    'Decision Tree'       : (dt,      False),
    'Random Forest'       : (rf,      False),
    'Naive Bayes'         : (nb,      False),
    'Gradient Boosting'   : (gb,      False),
    'AdaBoost'            : (ada,     False),
    'XGBoost Classifier'  : (xgb_c,  False),
    'CatBoost Classifier' : (cat_c,  False),
    'Bagging Classifier'  : (bag_c,  False),
    'Stacking Classifier' : (stack_c, True),
}

best_reg_name = res_df.iloc[0]['Model']
best_clf_name = res_clf_df.iloc[0]['Model']

best_reg_obj, reg_scaled = model_map_reg[best_reg_name]
best_clf_obj, clf_scaled = model_map_clf[best_clf_name]

joblib.dump(best_reg_obj, os.path.join(MODELS_DIR, "best_reg_model.pkl"))
joblib.dump(best_clf_obj, os.path.join(MODELS_DIR, "best_clf_model.pkl"))
joblib.dump({
    'reg_name'  : best_reg_name,
    'clf_name'  : best_clf_name,
    'reg_scaled': reg_scaled,
    'clf_scaled': clf_scaled,
    'best_r2'   : float(res_df.iloc[0]['R²']),
    'best_acc'  : float(res_clf_df.iloc[0]['Accuracy']),
}, os.path.join(MODELS_DIR, "meta.pkl"))

print(f"Saved best regression model  : {best_reg_name}")
print(f"Saved best classification model: {best_clf_name}")
```

---

## Cell D — Copy cleaned dataset to API folder

```python
import shutil
API_DIR   = r"C:\path\to\PakWheelsAPI"   # ← same folder as main.py
shutil.copy("PW_dataset.csv", os.path.join(API_DIR, "PW_dataset.csv"))
print("Dataset copied to API folder.")
```

---

## Cell E — POST all regression results to API (auto-fills the dashboard)

```python
for _, row in res_df.iterrows():
    payload = {
        "model_name": row["Model"],
        "r2"        : float(row["R²"]),
        "rmse_log"  : float(row["RMSE(log)"]),
        "mae_log"   : float(row["MAE(log)"]),
        "rmse_lacs" : float(row["RMSE(Lacs)"]),
        "mae_lacs"  : float(row["MAE(Lacs)"]),
    }
    r = requests.post(f"{API_URL}/api/results/regression", json=payload)
    print(f"{row['Model']:30s} → {r.status_code}")
```

---

## Cell F — POST all classification results to API

```python
for _, row in res_clf_df.iterrows():
    payload = {
        "model_name": row["Model"],
        "accuracy"  : float(row["Accuracy"]),
        "f1_score"  : float(row["F1"]),
        "precision" : float(row["Precision"]),
        "recall"    : float(row["Recall"]),
    }
    r = requests.post(f"{API_URL}/api/results/classification", json=payload)
    print(f"{row['Model']:30s} → {r.status_code}")
```

---

## Running the API

```bash
cd C:\path\to\PakWheelsAPI
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs
