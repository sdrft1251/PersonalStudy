import cx_Oracle
import pandas as pd
import json
import requests
import config
from datetime import datetime
import exception_list
import os

class nppms_data_upload:
    def __init__(self, oracle_lib_path):
        cx_Oracle.init_oracle_client(lib_dir=oracle_lib_path)

    def connect_to_db(self, db_ip):
        try:
            conn = cx_Oracle.connect('@@','@@', db_ip + ':@@/ORCL')
            return conn
        except:
            raise exception_list.connection_error()

    def request_to_db(self, query, connect):
        df=pd.read_sql(query, con = connect)
        return df

    def get_weight_data(self, start_date, db_ip):
        con_result = self.connect_to_db(db_ip)
        try:
            df = self.request_to_db(query = "SELECT * FROM @@ WHERE @@ >= " +  start_date + "000000 ORDER BY @@ asc",
                                    connect = self.connect_to_db(db_ip))
            return df
        except:
            raise exception_list.get_weight_data_fail()

    def return_pdis_form(self, df):
        try:
            df['@@'] = ""
            df = df[config.column_format_list]
            df.fillna("",inplace=True)
            return df
        except:
            raise exception_list.transform_error()

    def upload_to_pdis(self, start_date, db_ip, save_opt=False):
        df = self.return_pdis_form(df=self.get_weight_data(start_date=start_date, db_ip=db_ip))
        try:
            if save_opt:
                df.to_excel('./log/{}_{}.xlsx'.format(config.weight_program_pc_reverse[db_ip], datetime.now().strftime("%y_%m_%d")))
        except:
            raise exception_list.save_error()

        try:
            headers = {'Content-Type': 'application/json; chearset=utf-8'}
            error_num=0
            error_info=[]

            for df_idx in range(df.shape[0]):
                dict_row = df.iloc[df_idx].to_dict()
                res = requests.post('@@', data=json.dumps(dict_row), headers=headers)
                if res.status_code != 200:
                    error_num += 1
                    error_info.append(dict_row['@@'])
                    print(dict_row['@@'])
                #######################################################################################################################
                #print(dict_row)
                #break
                #######################################################################################################################
        except:
            raise exception_list.connect_to_pdis()
        return error_num, error_info


def upload_all(start_date):
    def print_log(error_opt, name, ip, e=""):
        if error_opt:
            print("{} / {} : Faield to upload / --> {}".format(name, ip, e))
        else:
            print("{} / {} : Success to upload".format(name, ip))
    nppms_data_upload_ob = nppms_data_upload(oracle_lib_path=r"C:\instantclient_19_9_32")
    for name, ip in config.weight_program_pc.items():
        try:
            error_num, error_info = nppms_data_upload_ob.upload_to_pdis(start_date=start_date, db_ip=ip, save_opt=True)
            print_log(error_opt=False, name=name, ip=ip)
            #######################################################################################################################
            #break
            #######################################################################################################################
        except exception_list.connection_error as e:
            print_log(error_opt=True, name=name, ip=ip, e=e)
        except exception_list.get_weight_data_fail as e:
            print_log(error_opt=True, name=name, ip=ip, e=e)
        except exception_list.transform_error as e:
            print_log(error_opt=True, name=name, ip=ip, e=e)
        except exception_list.save_error as e:
            print_log(error_opt=True, name=name, ip=ip, e=e)
        except exception_list.connect_to_pdis as e:
            print_log(error_opt=True, name=name, ip=ip, e=e)

def connection_check():
    nppms_data_upload_ob = nppms_data_upload(oracle_lib_path=r"C:\instantclient_19_9_32")
    for name, ip in config.weight_program_pc.items():
        try:
            nppms_data_upload_ob.connect_to_db(db_ip=ip)
            print(name + " / " + ip + " : Connection success")
        except exception_list.connection_error as e:
            print(name + " / " + ip + " : Connection failed")

def last_data_check(start_date):
    nppms_data_upload_ob = nppms_data_upload(oracle_lib_path=r"C:\instantclient_19_9_32")
    for name, ip in config.weight_program_pc.items():
        try:
            df = nppms_data_upload_ob.get_weight_data(start_date=start_date, db_ip=ip)
            print(name + " / " + ip + " : DATA is ==>")
            print(df[["@@","@@"]].tail(10))
        except exception_list.connection_error as e:
            print(name + " / " + ip + " : Connection failed")
        except exception_list.get_weight_data_fail as e:
            print(name + " / " + ip + " : Getting data failed")

if __name__ == "__main__":
    upload_all(start_date="@@")
    #last_data_check(start_date="@@")
    #connection_check()

        
    
