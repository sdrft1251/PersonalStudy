


def q_check_table_exist(tablename):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tablename}';"
    return query

def q_get_table_names():
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    return query


def q_get_last_date(tablename):
    query = f"SELECT Date FROM '{tablename}' ORDER BY Date DESC LIMIT 1;"
    return query

def q_get_raw_data(tablename):
    query = f"SELECT * FROM '{tablename}' ORDER BY Date ASC;"
    return query


def q_create_table(table_name):
    query = f"""\
CREATE TABLE '{table_name}'
(
    Date TEXT PRIMARY KEY,
    Open INTEGER,
    High INTEGER,
    Low INTEGER,
    Close INTEGER,
    Volume INTEGER
);"""
    return query


def q_create_process_1_table(table_name):
    query = f"""\
CREATE TABLE '{table_name}'
(
    Date TEXT PRIMARY KEY,
    MA_High_5 REAL,
    MA_High_10 REAL,
    MA_High_20 REAL,
    MA_High_60 REAL,
    MA_High_120 REAL,
    MA_Low_5 REAL,
    MA_Low_10 REAL,
    MA_Low_20 REAL,
    MA_Low_60 REAL,
    MA_Low_120 REAL,
    MA_Close_5 REAL,
    MA_Close_10 REAL,
    MA_Close_20 REAL,
    MA_Close_60 REAL,
    MA_Close_120 REAL,
    MA_Volume_5 REAL,
    MA_Volume_10 REAL,
    MA_Volume_20 REAL,
    MA_Volume_60 REAL,
    MA_Volume_120 REAL,
    Highest_High_5 REAL,
    Highest_High_10 REAL,
    Highest_High_20 REAL,
    Highest_High_60 REAL,
    Highest_High_120 REAL,
    Highest_Low_5 REAL,
    Highest_Low_10 REAL,
    Highest_Low_20 REAL,
    Highest_Low_60 REAL,
    Highest_Low_120 REAL,
    Highest_Close_5 REAL,
    Highest_Close_10 REAL,
    Highest_Close_20 REAL,
    Highest_Close_60 REAL,
    Highest_Close_120 REAL,
    Highest_Volume_5 REAL,
    Highest_Volume_10 REAL,
    Highest_Volume_20 REAL,
    Highest_Volume_60 REAL,
    Highest_Volume_120 REAL,
    Lowest_High_5 REAL,
    Lowest_High_10 REAL,
    Lowest_High_20 REAL,
    Lowest_High_60 REAL,
    Lowest_High_120 REAL,
    Lowest_Low_5 REAL,
    Lowest_Low_10 REAL,
    Lowest_Low_20 REAL,
    Lowest_Low_60 REAL,
    Lowest_Low_120 REAL,
    Lowest_Close_5 REAL,
    Lowest_Close_10 REAL,
    Lowest_Close_20 REAL,
    Lowest_Close_60 REAL,
    Lowest_Close_120 REAL,
    Lowest_Volume_5 REAL,
    Lowest_Volume_10 REAL,
    Lowest_Volume_20 REAL,
    Lowest_Volume_60 REAL,
    Lowest_Volume_120 REAL,
    Target REAL
);"""
    return query