import os
import json
from models import XMLToMetaDB
from config import Config

db_base_path = Config.db_base_path


if __name__ == "__main__":
    for year in ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019", "2020", "2021"]:
        print(f"{year} is started")

        if year == "2020" or year == "2021":
            root_path = "/home/weladmin/Desktop/smc_data"
            db_path = os.path.join(db_base_path, f"smc_metadata_v04.db")
        else:
            root_path = "/data/smc_data"
            db_path = os.path.join(db_base_path, f"smc_metadata_v04.db")

        xml_to_db = XMLToMetaDB(db_path, root_path)
        xml_to_db.db_connect()
        failed_logs = xml_to_db.parsing(year)
        xml_to_db.db_close()

        print(f"Failed parsing num is : {len(failed_logs['parsing_failed'])}")
        print(f"Failed parsing num is : {len(failed_logs['save_failed'])}")
        with open(f"/data/datadisk/wellysis/smc/logs/failed_xmlparsing_{year}.json", "w") as f:
            json.dump(failed_logs, f)
