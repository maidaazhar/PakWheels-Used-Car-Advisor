import os

DB_SERVER         = r"localhost\SQLEXPRESS"
DB_NAME           = "PakWheelsDashboard"
CONNECTION_STRING = (
    f"DRIVER={{SQL Server}};SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};Trusted_Connection=yes;"
)
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH   = os.path.join(os.path.dirname(__file__), "PW_dataset.csv")
API_BASE    = "http://localhost:8000"
