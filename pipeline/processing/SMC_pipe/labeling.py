import os
import json
from models import ClassBinary, Labeling
from config import Config

db_base_path = Config.db_base_path


if __name__ == "__main__":
    db_path = os.path.join(db_base_path, f"smc_metadata_v04.db")
    # class_binary = ClassBinary(db_path)
    # class_binary.db_connect()
    # class_binary.labeling()
    # class_binary.db_close()

    label = Labeling(db_path)
    label.db_connect()
    label.labeling()
    label.db_close()

