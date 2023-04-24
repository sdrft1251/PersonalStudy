

def q_create_metadb_table(tablename):
    query = f"""\
CREATE TABLE {tablename}
(
    id INTEGER PRIMARY KEY,
    patient_id TEXT,
    birthdate TEXT,
    gender TEXT,
    start_datetime TEXT,
    sampling_rate TEXT,
    amplitude TEXT,
    diag_1 TEXT,
    diag_2 TEXT,
    diag_3 TEXT,
    diag_4 TEXT,
    diag_5 TEXT,
    diag_6 TEXT,
    diag_7 TEXT,
    diag_8 TEXT,
    diag_9 TEXT,
    diag_10 TEXT,
    diag_11 TEXT,
    diag_12 TEXT,
    diag_13 TEXT,
    diag_14 TEXT,
    diag_15 TEXT,
    diag_16 TEXT,
    diag_17 TEXT,
    diag_18 TEXT,
    diag_19 TEXT,
    diag_20 TEXT,
    wave_1_name TEXT,
    wave_2_name TEXT,
    wave_3_name TEXT,
    wave_4_name TEXT,
    wave_5_name TEXT,
    wave_6_name TEXT,
    wave_7_name TEXT,
    wave_8_name TEXT,
    wave_9_name TEXT,
    wave_10_name TEXT,
    wave_11_name TEXT,
    wave_12_name TEXT,
    origin_path TEXT
);"""
    return query


def q_check_table_exist(tablename):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tablename}';"
    return query


def make_diagnosis_to_query(diagnosis_list):
    diagnosis_str_for_query = ""
    for diag in diagnosis_list:
        diagnosis_str_for_query = diagnosis_str_for_query + f""", '{diag.replace("'","''")}'"""
    return diagnosis_str_for_query

def make_wave_to_query(wave_name_list):
    wave_str_for_query = ""
    for name in wave_name_list:
        wave_str_for_query = wave_str_for_query+ f", '{name}'"
    return wave_str_for_query

                
def q_insert_to_metadb(tablename, index_val, id_, birthdate, gender, start_datetime, sampling_rate, amplitude, diagnosis_list, wave_name_list, xml_path):
    diagnosis_str_for_query = make_diagnosis_to_query(diagnosis_list)
    wave_str_for_query = make_wave_to_query(wave_name_list)
    query = f"""\
INSERT INTO {tablename} VALUES \
({index_val}, '{id_}', '{birthdate}', '{gender}', '{start_datetime}', '{sampling_rate}', '{amplitude}'\
{diagnosis_str_for_query}{wave_str_for_query}, '{xml_path}');"""
    return query


def q_get_last_index(tablename):
    query = f"SELECT id FROM {tablename} ORDER BY id DESC LIMIT 1;"
    return query

def q_check_record_exist(table_name, xml_path):
    query = f"SELECT id FROM {table_name} WHERE origin_path='{xml_path}';"
    return query


def create_wave_table(tablename):
    query = f"""\
CREATE TABLE {tablename}
(
    origin_path TEXT PRIMARY KEY,
    sampling_rate TEXT,
    filename TEXT
);"""
    return query