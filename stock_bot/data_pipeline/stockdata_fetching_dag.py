import os
import datetime

import airflow
from airflow import DAG


from data_pipeline.operators import StockDataHandler, ProcessingForModel_1
from data_pipeline.config import Config

dag = DAG(
    dag_id="stockdata_fetching_dag",
    start_date=airflow.utils.dates.days_ago(1),
    schedule_interval="0 15 * * *",
)

get_stock_data = StockDataHandler(
    task_id="get_stock_data",
    db_path=Config.KOSDAQ_ORIGIN_DB_PATH,
    target_market="KOSDAQ",
    end_date=None,
    dag=dag,
)

processing_1 = ProcessingForModel_1(
    task_id="processing_1",
    raw_db_path=Config.KOSDAQ_ORIGIN_DB_PATH,
    processed_input_db_path=Config.KOSDAQ_PREPROCESS_1_DB_PATH,
    processed_predict_db_path=Config.KOSDAQ_PREPROCESS_1_VAL_DB_PATH,
    dag=dag,
)

get_stock_data >> processing_1