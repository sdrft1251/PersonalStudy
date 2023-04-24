import os
import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd

from parser import philips_parser, past_parser
from utils import sql, merge_table, replace_datetime


class XMLToMetaDB:
    def __init__(self, metadb_path, origin_data_path):
        self.metadb_path = metadb_path
        self.origin_data_path = origin_data_path
        self.metadb_conn = None

    
    def db_connect(self):
        if self.metadb_conn is None:
            self.metadb_conn = sqlite3.connect(self.metadb_path)

    def db_close(self):
        if self.metadb_conn is not None:
            self.metadb_conn.close()

    def create_table(self, tablename):
        cur = self.metadb_conn.cursor()
        cur.execute(sql.q_create_metadb_table(tablename))
        self.metadb_conn.commit()
        cur.close()

    def get_xml_dirs(self, path):
        file_dirs = []
        for root, dirs, files in os.walk(path):
            if len(files)>0:
                for file_name in files:
                    if file_name.endswith('.xml'):
                        file_dirs.append(os.path.join(root, file_name))
        return file_dirs
    
    def check_table_exist(self, table_name):
        cur = self.metadb_conn.cursor()
        cur.execute(sql.q_check_table_exist(table_name))
        results = cur.fetchall()
        if len(results) == 0:
            cur.close()
            return 0
        else:
            cur.execute(sql.q_get_last_index(table_name))
            index_val = cur.fetchall()[0][0]
            cur.close()
            return index_val
        

    def reformating_diagnosis_list(self, diagnosis_list):
        # Not enough
        while(len(diagnosis_list)<20):
            diagnosis_list.append("None")
        # If exceed
        if len(diagnosis_list)>20:
            new_list = []
            reduce_num = len(diagnosis_list)-20
            merge_str = ""
            for idx, val in enumerate(diagnosis_list):
                if idx<19:
                    new_list.append(val)
                else:
                    merge_str += f"___{val}"
            new_list.append(merge_str)
            diagnosis_list = new_list
        return diagnosis_list

    def check_record_exist(self, table_name, xml_path):
        cur = self.metadb_conn.cursor()
        cur.execute(sql.q_check_record_exist(table_name, xml_path))
        result = cur.fetchall()
        if len(result) != 0:
            return 1
        else:
            return 0
    
    def parsing(self, year):
        print(f"Parsing {year} -> start!")

        data_path = os.path.join(self.origin_data_path, year)
        xml_dirs = self.get_xml_dirs(data_path)

        # Create Table
        table_name = f"patientinfo__{year}"
        check_val = self.check_table_exist(table_name)
        if check_val == 0:
            self.create_table(table_name)
            index_val = 1
        else:
            index_val = check_val+1
        
        # Works Load
        total_to_do = len(xml_dirs)
        done = 0
        # Init
        parsing_failed = []
        save_failed = []
        for xml_path in xml_dirs:
            check_record = self.check_record_exist(table_name, xml_path)
            if check_record == 1:
                done += 1
                continue 
            # Parsing
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                # Type 구분
                patient_info = root.find("PatientInfo")
                for_type = patient_info.find("PatientID")
                if for_type is None:   # Type -> Philips
                    obj = philips_parser(root=root)
                    # Get Info
                    id_ = obj.get_id()
                    birthdate = obj.get_birthdate()
                    gender = obj.get_gender()
                    start_datetime = obj.get_startdatetime()
                    diagnosis_list = obj.get_diagnosis()
                    sampling_rate = obj.get_sampling_rate()
                    amplitude = obj.get_amplitude()
                    wave_name_list = obj.get_wavename_list()

                else:   # Type -> B Type
                    obj = past_parser(root=root)
                    # Get Info
                    id_ = obj.get_id()
                    birthdate = obj.get_birthdate(xml_path=xml_path)
                    gender = obj.get_gender()
                    start_datetime = obj.get_startdatetime(xml_path=xml_path)
                    diagnosis_list = obj.get_diagnosis()
                    sampling_rate = obj.get_sampling_rate()
                    amplitude = obj.get_amplitude()
                    wave_name_list = obj.get_wavename_list()

            except Exception as e:
                print(f"{xml_path} Reading Failed!!! --> {e}")
                parsing_failed.append(xml_path)
                done += 1
                continue
            
            # Reformating diagnosis
            diagnosis_list = self.reformating_diagnosis_list(diagnosis_list)

            try:
                cur = self.metadb_conn.cursor()
                # Input to DB
                cur.execute(sql.q_insert_to_metadb(table_name, index_val, id_, birthdate, gender, start_datetime, sampling_rate, amplitude, diagnosis_list, wave_name_list, xml_path))
                self.metadb_conn.commit()
                cur.close()
                index_val+=1
            except Exception as e:
                print(f"{xml_path} Input Failed!!! --> {e}")
                save_failed.append(xml_path)
        
        done += 1
        if done % 10000 == 0:
            done_rate = (done/total_to_do)*100
            print(f"{year} -> Done rate : {done_rate: 2.2f} | Done Index : {done}")

        failed_logs = {
            "parsing_failed": parsing_failed,
            "save_failed": save_failed
        }
        return failed_logs
                

class LinkPatient:
    def __init__(self, metadb_path):
        self.metadb_path = metadb_path
        self.metadb_conn = None
    
    def db_connect(self):
        if self.metadb_conn is None:
            self.metadb_conn = sqlite3.connect(self.metadb_path)

    def db_close(self):
        if self.metadb_conn is not None:
            self.metadb_conn.close()

    def list_to_str(self, data_list, split=","):
        if len(data_list) == 0:
            return ""
        base = ""
        for data in data_list:
            base += f",{data}"
        return base[1:]

    def link(self, columns_list=["patient_id","start_datetime","origin_path"], year_list=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019", "2020", "2021"]):
        merge_data = merge_table(columns_list, year_list, self.metadb_conn)
        merge_data["record_id"] = merge_data["origin_path"]
        print(merge_data.shape)
        merge_data["start_datetime"] = merge_data["start_datetime"].apply(replace_datetime)

        total_to_do = len(merge_data["patient_id"].unique())
        done = 0
        patient_tracking = pd.DataFrame(columns=["patient_id", "record_num", "record_id_seq"])

        patient_dump = []
        record_num_dump = []
        record_id_seq_dump = []

        for patient_id, slice_df in merge_data.groupby(merge_data["patient_id"]):
            slice_df = slice_df.sort_values(by=["start_datetime"], axis=0)[["record_id", "start_datetime"]]
            record_id_list = list(slice_df["record_id"].values)
            record_id_seq = self.list_to_str(record_id_list)
            patient_dump.append(patient_id)
            record_num_dump.append(len(record_id_list))
            record_id_seq_dump.append(record_id_seq)

            done += 1
            if done % 10000 == 0:
                done_rate = (done/total_to_do)*100
                print(f"Done rate : {done_rate: 2.2f} | Done Index : {done}")

        patient_tracking["patient_id"] = patient_dump
        patient_tracking["record_num"] = record_num_dump
        patient_tracking["record_id_seq"] = record_id_seq_dump
        
        patient_tracking.to_sql('patient_tracking', self.metadb_conn, index=False)


class ClassBinary:
    def __init__(self, metadb_path):
        self.metadb_path = metadb_path
        self.metadb_conn = None
        self.diag_col_name_list = ["diag_1", "diag_2", "diag_3", "diag_4", "diag_5", "diag_6", "diag_7", "diag_8", "diag_9", "diag_10", "diag_11", "diag_12", "diag_13", "diag_14", "diag_15", "diag_16", "diag_17", "diag_18", "diag_19", "diag_20"]
    
    def db_connect(self):
        if self.metadb_conn is None:
            self.metadb_conn = sqlite3.connect(self.metadb_path)

    def db_close(self):
        if self.metadb_conn is not None:
            self.metadb_conn.close()
    

    def labeling(self,\
    columns_list=["patient_id","start_datetime","origin_path", "diag_1", "diag_2", "diag_3", "diag_4", "diag_5", "diag_6", "diag_7", "diag_8", "diag_9", "diag_10", "diag_11", "diag_12", "diag_13", "diag_14", "diag_15", "diag_16", "diag_17", "diag_18", "diag_19", "diag_20"],\
    year_list=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019", "2020", "2021"]):
        merge_data = merge_table(columns_list, year_list, self.metadb_conn)
        merge_data["record_id"] = merge_data["origin_path"]
        print(merge_data.shape)
        merge_data["start_datetime"] = merge_data["start_datetime"].apply(replace_datetime)

        def normal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "normal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def abnormal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "abnormal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def boderline_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "borderline ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def otherwise_normal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "otherwise normal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        merge_data["normal_ecg"] = merge_data.apply(normal_ecg_check, axis=1)
        merge_data["abnormal_ecg"] = merge_data.apply(abnormal_ecg_check, axis=1)
        merge_data["borderline_ecg"] = merge_data.apply(boderline_ecg_check, axis=1)
        merge_data["otherwise_normal_ecg"] = merge_data.apply(otherwise_normal_ecg_check, axis=1)

        need_col = ["patient_id","start_datetime","record_id","normal_ecg","abnormal_ecg","borderline_ecg","otherwise_normal_ecg"]
        result_df = merge_data[need_col]

        result_df.to_sql('normal_abnormal_label', self.metadb_conn, index=False)

        self.metadb_conn.commit()


class Labeling:
    def __init__(self, metadb_path):
        self.metadb_path = metadb_path
        self.metadb_conn = None
        self.diag_col_name_list = ["diag_1", "diag_2", "diag_3", "diag_4", "diag_5", "diag_6", "diag_7", "diag_8", "diag_9", "diag_10", "diag_11", "diag_12", "diag_13", "diag_14", "diag_15", "diag_16", "diag_17", "diag_18", "diag_19", "diag_20"]
    
    def db_connect(self):
        if self.metadb_conn is None:
            self.metadb_conn = sqlite3.connect(self.metadb_path)

    def db_close(self):
        if self.metadb_conn is not None:
            self.metadb_conn.close()
    

    def labeling(self,\
    columns_list=["patient_id","start_datetime","origin_path", "diag_1", "diag_2", "diag_3", "diag_4", "diag_5", "diag_6", "diag_7", "diag_8", "diag_9", "diag_10", "diag_11", "diag_12", "diag_13", "diag_14", "diag_15", "diag_16", "diag_17", "diag_18", "diag_19", "diag_20"],\
    year_list=["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019", "2020", "2021"]):
        merge_data = merge_table(columns_list, year_list, self.metadb_conn)
        merge_data["record_id"] = merge_data["origin_path"]
        print(merge_data.shape)
        merge_data["start_datetime"] = merge_data["start_datetime"].apply(replace_datetime)

        def check_afib_first(raw):
            for col in self.diag_col_name_list:
                val =  raw[col].strip().lower()
                if "atrial fibrillation" in val:
                    return 1
                elif "atrial" in val and "fib" in val:
                    return 1
                elif "a-flutter/fibrillation" in val:
                    return 1
                elif 'afib' in val:
                    return 1
            return 0

        def normal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "normal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def abnormal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "abnormal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def boderline_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "borderline ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        def otherwise_normal_ecg_check(raw):
            for col in self.diag_col_name_list:
                if "otherwise normal ecg" == raw[col].replace("-","").strip().lower():
                    return 1
            return 0

        merge_data["normal_ecg"] = merge_data.apply(normal_ecg_check, axis=1)
        merge_data["abnormal_ecg"] = merge_data.apply(abnormal_ecg_check, axis=1)
        merge_data["borderline_ecg"] = merge_data.apply(boderline_ecg_check, axis=1)
        merge_data["otherwise_normal_ecg"] = merge_data.apply(otherwise_normal_ecg_check, axis=1)
        merge_data["afib"] = merge_data.apply(check_afib_first, axis=1)

        need_col = ["patient_id","start_datetime","record_id","normal_ecg","abnormal_ecg","borderline_ecg","otherwise_normal_ecg","afib"]
        result_df = merge_data[need_col]

        result_df.to_sql('label', self.metadb_conn, index=False)

        self.metadb_conn.commit()

class XMLSignalToNumpy:
    def __init__(self, metadb_path, origin_data_path):
        self.metadb_path = metadb_path
        self.origin_data_path = origin_data_path
        self.metadb_conn = None

    def db_connect(self):
        self.metadb_conn = sqlite3.connect(self.metadb_path)

    def db_close(self):
        if self.metadb_conn is not None:
            self.metadb_conn.close()

    def get_xml_dirs(self, path):
        file_dirs = []
        for root, dirs, files in os.walk(path):
            if len(files)>0:
                for file_name in files:
                    if file_name.endswith('.xml'):
                        file_dirs.append(os.path.join(root, file_name))
        return file_dirs

    def check_table_exist(self, table_name):
        cur = self.metadb_conn.cursor()
        cur.execute(sql.q_check_table_exist(table_name))
        results = cur.fetchall()
        if len(results) == 0:
            cur.close()
            return 0

    def parsing(self, lead, output_path):
        print(f"Parsing Signal {lead} -> start!")
        # For Output
        os.makedirs(output_path, exist_ok=True)


        xml_dirs = self.get_xml_dirs(self.origin_data_path)

        # for xml_path in xml_dirs:
        


    




