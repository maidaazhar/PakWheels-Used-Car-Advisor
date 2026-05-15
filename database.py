import pyodbc
from config  import CONNECTION_STRING
from schemas import RegressionResultIn, ClassificationResultIn

def get_conn():
    return pyodbc.connect(CONNECTION_STRING)

# ── Regression ────────────────────────────────────────────────────────────────
def db_insert_regression(r: RegressionResultIn):
    sql = """INSERT INTO RegressionResults
             (ModelName,R2,RmseLog,MaeLog,RmseLacs,MaeLacs)
             VALUES (?,?,?,?,?,?)"""
    with get_conn() as conn:
        conn.execute(sql,(r.model_name,r.r2,r.rmse_log,r.mae_log,r.rmse_lacs,r.mae_lacs))
        conn.commit()

def db_get_all_regression():
    with get_conn() as conn:
        cur  = conn.execute("SELECT * FROM RegressionResults ORDER BY R2 DESC")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols,row)) for row in cur.fetchall()]

def db_delete_regression(id:int):
    with get_conn() as conn:
        conn.execute("DELETE FROM RegressionResults WHERE Id=?",(id,))
        conn.commit()

# ── Classification ────────────────────────────────────────────────────────────
def db_insert_classification(r: ClassificationResultIn):
    sql = """INSERT INTO ClassificationResults
             (ModelName,Accuracy,F1Score,PrecisionScore,Recall)
             VALUES (?,?,?,?,?)"""
    with get_conn() as conn:
        conn.execute(sql,(r.model_name,r.accuracy,r.f1_score,r.precision,r.recall))
        conn.commit()

def db_get_all_classification():
    with get_conn() as conn:
        cur  = conn.execute("SELECT * FROM ClassificationResults ORDER BY Accuracy DESC")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols,row)) for row in cur.fetchall()]

def db_delete_classification(id:int):
    with get_conn() as conn:
        conn.execute("DELETE FROM ClassificationResults WHERE Id=?",(id,))
        conn.commit()
