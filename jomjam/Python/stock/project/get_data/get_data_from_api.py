import win32com.client
import pandas as pd
import time
import sqlite3

instCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
print(instCpCybos.IsConnect)

instCpCodeMgr = win32com.client.Dispatch("CpUtil.CpCodeMgr")
codeList = instCpCodeMgr.GetStockListByMarket(1)
print(codeList)
instStockChart = win32com.client.Dispatch("CpSysDib.StockChart")



stockchart_params={
    'date':0,
    'time':1,
    'open':2,
    'high':3,
    'low':4,
    'close':5,
    'increase_amount':6,
    'volume':8,
    'total_num_of_stock':12,
    'market_cap':13,
    'forigner_have_rate':17,
    'corp_net_buy':20,
    'corp_cum_net_buy':21
}
col_key_list=list(stockchart_params.keys())

for code in codeList[:2]:
    instStockChart.SetInputValue(0,code)
    instStockChart.SetInputValue(1, ord('2'))
    instStockChart.SetInputValue(4, 10000000)
    instStockChart.SetInputValue(5, (0,2,3,4,5,6,8,12,13,17,20,21))
    instStockChart.SetInputValue(6, ord('D'))
    instStockChart.SetInputValue(9, ord('1'))
    instStockChart.BlockRequest()

    numData = instStockChart.GetHeaderValue(3)
    numField = instStockChart.GetHeaderValue(1)

    result_dict={}
    for col_idx in range(numField):
        result_dict[col_key_list[col_idx]] = []
        for day_idx in range(numData):
            result_dict[col_key_list[col_idx]].append(instStockChart.GetDataValue(col_idx, day_idx))

    tmp_df = pd.DataFrame(result_dict, columns=col_key_list)
    conn = sqlite3.connect("kospi_past_day_data_set.db")
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM " + code)
        rows = cur.fetchall()
        cols = [column[0] for column in cur.description]
        reload_df = pd.DataFrame.from_records(data=rows, columns=cols)
        last_date_from_save = reload_df.date.iloc[0]
        tmp_df = tmp_df[tmp_df.date>last_date_from_save]
        tmp_df = pd.concat([tmp_df, reload_df], ignore_index=True, axis=0)
    except:
        print("There is no table")
    tmp_df.to_sql(code, con=conn, if_exists='replace', index=False)
    time.sleep(1)



