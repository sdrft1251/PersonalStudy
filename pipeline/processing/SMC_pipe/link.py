import os
import json
from models import LinkPatient
from config import Config

db_base_path = Config.db_base_path

if __name__ == "__main__":
    db_path = os.path.join(db_base_path, f"smc_metadata_v04.db")
    link = LinkPatient(db_path)
    link.db_connect()
    link.link()
    link.db_close()

