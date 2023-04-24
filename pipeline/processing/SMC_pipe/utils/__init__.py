import pandas as pd
import datetime

# For Merging
def merge_table(columns_list, year_list, con):
    base = ""
    for col in columns_list:
        base = base+f", {col}"
    columns = base[1:]
    merged = pd.DataFrame(columns=columns_list)
    for year_ in year_list:
        meta = pd.read_sql(f"SELECT{columns} FROM patientinfo__{year_}", con)
        merged = pd.concat([merged, meta], ignore_index=True, axis=0)
    return merged

def replace_datetime(raw):
    return datetime.datetime.strptime(raw, '%Y-%m-%d_%H:%M:%S')