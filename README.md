# 🚗 PakWheels Used Car Advisor

A full-stack machine learning application that predicts used car prices and classifies cars into budget tiers using ~45,000 real listings scraped from [PakWheels.com](https://www.pakwheels.com). Built as a semester project for **Machine Learning for Business Analytics (Spring 2026)**.

---

## 📌 What It Does

- **Price Prediction** — regression models estimate a car's market value in PKR
- **Tier Classification** — classifies any car as Budget, Mid-Range, or Luxury
- **Cluster Analysis** — unsupervised learning reveals hidden market segments
- **Web App** — an ASP.NET dashboard where users enter car specs and get instant predictions

---

## 🗂️ Repository Structure

```
PakWheels-Used-Car-Advisor/
│
├── Notebook/
│   └── PakWheels_ML.ipynb          # Full ML pipeline (EDA → models → clustering)
│
├── PakWheelsAPI/                   # FastAPI backend serving trained models
│   ├── main.py
│   ├── requirements.txt
│   └── models/                     # Saved .pkl model files
│
├── PakWheelsPredictApp/            # ASP.NET Core frontend dashboard
│   ├── Controllers/
│   ├── Views/
│   ├── Models/
│   └── PakWheelsPredictApp.csproj
│
└── Data/
    └── pakwheels_listings.csv      # Raw dataset (~45,000 rows)
```

---

## 🧪 ML Pipeline (Jupyter Notebook)

### Data Preprocessing
- String cleaning on `Price`, `Mileage`, `Engine Capacity`
- Feature engineering: `Car Age`, `Mileage per Year`, `Feature Count`, `Brand Prestige Score`, `Engine Category`, `Is Imported`
- Domain-based outlier removal
- Square root transformation for skewed features (log over-corrected to skewness −3.79)
- Label encoding for categorical variables

### Regression Models (10+)
| Model | Notes |
|---|---|
| Linear Regression | Baseline, scaled input |
| Ridge / Lasso | Regularization variants |
| Polynomial Regression | Degree-2 features |
| Decision Tree | No scaling needed |
| Random Forest | Best single-model performer |
| Gradient Boosting | Ensemble boosting |
| AdaBoost | Adaptive boosting |
| Extra Trees | Randomized splits |
| XGBoost | Gradient boosting (optimized) |
| CatBoost | Handles categoricals natively |
| SVR | Kernel-based regression |
| KNN Regressor | Distance-based |
| Bagging Regressor | Bootstrap aggregating |
| Stacking Regressor | Meta-learner ensemble |
| Bayesian Ridge | Probabilistic linear model |

### Classification Models (10+)
Price tiers defined by domain-based thresholds (Budget / Mid-Range / Luxury):

Logistic Regression · KNN · Decision Tree · Random Forest · SVM · Naive Bayes · Gradient Boosting · AdaBoost · Extra Trees · LDA · XGBoost · CatBoost · Bagging · Stacking

### Unsupervised Learning
| Method | Details |
|---|---|
| K-Means | Full dataset (~45K rows) |
| K-Medoids | Sampled to 2,000 rows (O(n²) memory constraint) |
| Agglomerative Clustering | Sampled to 2,000 rows (O(n²) memory constraint) |
| PCA | Dimensionality reduction for cluster visualization |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML & EDA | Python, scikit-learn, XGBoost, CatBoost, pandas, NumPy, matplotlib, seaborn |
| API | Python, FastAPI, Uvicorn |
| Frontend | ASP.NET Core (C#), HTML/CSS, ADO.NET, SQL Server |
| Notebook | Jupyter Notebook |

---

## 🚀 Running the App

### Prerequisites
- Python 3.9+
- .NET 8 SDK
- SQL Server (local instance)

### Terminal 1 — FastAPI Backend

```powershell
cd PakWheelsAPI
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

### Terminal 2 — ASP.NET Dashboard

```powershell
cd PakWheelsPredictApp
dotnet run
```

Dashboard available at: `http://localhost:5000`

---

## 📊 Dataset

- **Source:** Scraped from PakWheels.com
- **Size:** ~45,000 used car listings
- **Features:** Make, Model, Year, Mileage, Engine Capacity, Fuel Type, Transmission, Color, City, Registered In, Assembly, Price (PKR)
- **Download:** [Google Drive →](https://drive.google.com/drive/folders/1xI4rLHOHPW28isZ5bChxfpKxkVQ7snpf?usp=sharing)

> The CSV is not included in this repo due to file size. Download it from the link above and place it at `Data/pakwheels_listings.csv` before running the notebook.

---

## 🔑 Key Engineering Decisions

- **Square root over log transformation** — log over-corrected Mileage per Year skewness to −3.79; sqrt gave a cleaner distribution
- **Sampling for O(n²) algorithms** — K-Medoids and Agglomerative Clustering were run on 2,000-row samples to avoid ~15 GiB memory failures on the full dataset
- **Nullable Int64 fix** — scikit-learn rejects pandas `Int64` dtype; all engineered features cast to standard `float64`
- **Linear models get scaled data** — tree-based models receive raw features; linear/SVM/KNN models receive `StandardScaler` output

---

## 👩‍💻 Author

**Maida Azhar** — FAST NUCES BSBA student, Machine Learning for Business Analytics (Spring 2026)

---

## 📄 License

This project was built for academic purposes. Dataset belongs to PakWheels.com.
