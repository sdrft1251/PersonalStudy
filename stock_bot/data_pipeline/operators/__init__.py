from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from data_pipeline.operators.processing_module import process_1
from data_pipeline.operators.utils import get_stock_past_data, get_code_list, replace_datetime_to_string
from data_pipeline.operators import sql
import sqlite3
import pandas as pd
from datetime import datetime, timedelta


class StockDataHandler(BaseOperator):
    @apply_defaults
    def __init__(self, db_path, target_market, end_date=None, **kwargs):
        super(StockDataHandler, self).__init__(**kwargs)
        self.db_path = db_path
        self.target_market = target_market
        self.end_date = end_date
        self.conn = None

    def db_connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

    def db_close(self):
        if self.conn is None:
            pass
        else:
            self.conn.close()

    def get_stock_list(self, symbol):
        return get_code_list(symbol)

    def prepare_df(self, code):
        cur = self.conn.cursor()
        cur.execute(sql.q_check_table_exist(code))
        exist_table = cur.fetchall()
        if len(exist_table) == 0: # Need to Create Table
            cur.execute(sql.q_create_table(code))
            self.conn.commit()
            start_date = "1900-01-01"
        else: # Already Exist
            cur.execute(sql.q_get_last_date(code))
            last_data = cur.fetchall()
            if len(last_data) != 0:
                start_date = (datetime.strptime(last_data[0][0], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "1900-01-01"
        cur.close()
        return start_date
    
    def save_stockdata_from_stocklist(self, code_list, end_date=None):
        total_to_do = len(code_list)
        done = 0
        failed_list = []
        for code in code_list:
            start_date = self.prepare_df(code)
            data = get_stock_past_data(code, start_date, end_date).reset_index()
            if data.shape[0]!=0:
                try:
                    data['Date'] = data['Date'].apply(replace_datetime_to_string)
                    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    data.to_sql(code, con=self.conn, if_exists='append', index=False)
                except Exception as e:
                    print(f"Saving to SQL Failed -> {e} | Code -> {code}")
                    failed_list.append(code)
            else:
                failed_list.append(code)

            done += 1
            if done%100 == 0:
                print(f"Done rate : {(done/total_to_do)*100: 2.2f} | Index : {done}")
        self.conn.commit()
        return failed_list

    def execute(self, context):
        self.db_connect()

        code_list = self.get_stock_list(self.target_market)
        failed_list = self.save_stockdata_from_stocklist(code_list, self.end_date)

        print(f"Failed code Num -> {len(failed_list)}")
        print(f"List -> {failed_list}")

        self.db_close()

class ProcessingForModel_1(BaseOperator):
    @apply_defaults
    def __init__(self, raw_db_path, processed_input_db_path, processed_predict_db_path, **kwargs):
        super(ProcessingForModel_1, self).__init__(**kwargs)
        self.raw_db_path = raw_db_path
        self.processed_input_db_path = processed_input_db_path
        self.processed_predict_db_path = processed_predict_db_path
        self.raw_db_conn = None
        self.processed_input_db_conn = None
        self.processed_predict_db_conn = None

    def db_connect(self):
        if self.raw_db_conn is None:
            self.raw_db_conn = sqlite3.connect(self.raw_db_path)
        if self.processed_input_db_conn is None:
            self.processed_input_db_conn = sqlite3.connect(self.processed_input_db_path)
        if self.processed_predict_db_conn is None:
            self.processed_predict_db_conn = sqlite3.connect(self.processed_predict_db_path)

    def db_close(self):
        if self.raw_db_conn is None:
            pass
        else:
            self.raw_db_conn.close()
        if self.processed_input_db_conn is None:
            pass
        else:
            self.processed_input_db_conn.close()
        if self.processed_predict_db_conn is None:
            pass
        else:
            self.processed_predict_db_conn.close()

    def get_code_list_from_raw_db(self):
        raw_db_cur = self.raw_db_conn.cursor()
        raw_db_cur.execute(sql.q_get_table_names())
        table_names = raw_db_cur.fetchall()
        raw_db_cur.close()
        return table_names

    def processing(self, raw_db):
        processing_obj = process_1(raw_db)
        train_part, predict_part = processing_obj.prcossing_df()
        return train_part, predict_part

    
    def prepare_processed_df(self, conn, code):
        processed_db_cur = conn.cursor()
        processed_db_cur.execute(sql.q_check_table_exist(code))
        exist_table = processed_db_cur.fetchall()
        if len(exist_table) == 0: # Need to Create Table
            processed_db_cur.execute(sql.q_create_process_1_table(code))
            self.processed_input_db_conn.commit()
            start_date = "1900-01-01"
        else: # Already Exist
            processed_db_cur.execute(sql.q_get_last_date(code))
            last_data = processed_db_cur.fetchall()
            if len(last_data) != 0:
                start_date = (datetime.strptime(last_data[0][0], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "1900-01-01"
        processed_db_cur.close()
        return start_date


    def save_processed_data(self, target_codes):
        # Done rate
        total_to_do = len(target_codes)
        done = 0
        failed_list = []
        # processing one by one
        for code in target_codes:
            code = code[0]
            raw_data = pd.read_sql(sql.q_get_raw_data(code), self.raw_db_conn)
            train_part, predict_part = self.processing(raw_data)
            predict_part["Target"] = 0

            # check exist
            for_train_start_date = self.prepare_processed_df(self.processed_input_db_conn, code)
            for_predict_start_date = self.prepare_processed_df(self.processed_predict_db_conn, code)

            train_part_inputs = train_part[train_part["Date"]>=for_train_start_date]
            if train_part_inputs.shape[0] != 0:
                try:
                    train_part_inputs.to_sql(code, con=self.processed_input_db_conn, if_exists='append', index=False)
                except Exception as e:
                    print(f"Saving TrainDF to SQL Failed -> {e} | Code -> {code}")
                    failed_list.append(code)

            predict_part_inputs = predict_part[predict_part["Date"]>=for_predict_start_date]
            if predict_part_inputs.shape[0] != 0:
                try:
                    predict_part_inputs.to_sql(code, con=self.processed_predict_db_conn, if_exists='append', index=False)
                except Exception as e:
                    print(f"Saving PredictnDF to SQL Failed -> {e} | Code -> {code}")
                    failed_list.append(code)

            done += 1
            if done%100 == 0:
                print(f"Done rate : {(done/total_to_do)*100: 2.2f} | Index : {done}")
        return failed_list

    def execute(self, context):
        self.db_connect()
        target_codes = self.get_code_list_from_raw_db()
        failed_list = self.save_processed_data(target_codes)

        print(f"Failed code Num -> {len(failed_list)}")
        print(f"List -> {failed_list}")

        self.db_close()


