import os
import json
import pandas as pd
import sqlite3
from config import Config

db_base_path = Config.db_base_path


if __name__ == "__main__":
    db_path = os.path.join(db_base_path, f"smc_metadata_v04.db")
    conn = sqlite3.connect(db_path)

    df = pd.read_sql("SELECT * FROM patientinfo__2021;", conn)

    print(df.patient_id)
    conn.close()
