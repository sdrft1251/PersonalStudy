import sqlite3
from utils import config, tech_indicator
import pandas as pd


def return_default_df(code, trading_date_int, add_length):
    conn = sqlite3.connect(config.db_path)
    cur = conn.cursor()
    cur.execute( "SELECT * FROM " + code + " WHERE date<="+str(trading_date_int) + " LIMIT " + str(add_length+tech_indicator.have_to_cut_num) + ";")
    rows = cur.fetchall()
    cols = [column[0] for column in cur.description]
    reload_df = pd.DataFrame.from_records(data=rows, columns=cols)
    reload_df.sort_values(by=['date'], inplace=True)
    reload_df.reset_index(inplace=True, drop=True)

    cur.execute( "SELECT * FROM nasdaq_index;")
    rows = cur.fetchall()
    cols = [column[0] for column in cur.description]
    nasdaq_df = pd.DataFrame.from_records(data=rows, columns=cols)

    reload_df = pd.merge(reload_df, nasdaq_df, on='date', how='left')
    reload_df.fillna(method='ffill', inplace=True)
    conn.close()
    return reload_df
