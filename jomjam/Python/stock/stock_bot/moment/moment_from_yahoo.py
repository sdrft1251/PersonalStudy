import pandas_datareader
import pandas as pd
import datetime
from common import summary
from common import crawling_naver_finance

def historical_data(stock_code):
    stock_code_input = stock_code + ".KS"
    
    now = datetime.datetime.now()
    end_date = now.strftime('%Y-%m-%d')
    start = now - datetime.timedelta(days=365)
    start_date = start.strftime('%Y-%m-%d')

    google_data = pandas_datareader.DataReader(stock_code_input,'yahoo', start_date, end_date)
    return google_data

def finance_info(stock_code, chart_type="year"):
    df_called = crawling_naver_finance.return_balancetable(corp_code=stock_code, year_quarter=chart_type)
    per = df_called["PER(배)"]
    eps = df_called["EPS(원)"]
    roe = df_called["ROE(%)"]
    result_str = "PER / EPS / ROE \n"
    for i in range(df_called.shape[0]):
        result_str = result_str + df_called.index.values[i] + " * "
        result_str = result_str + df_called["PER(배)"].values[i] + " * "
        result_str = result_str + df_called["EPS(원)"].values[i] + " * "
        result_str = result_str + df_called["ROE(%)"].values[i] + "\n"
    return result_str

def summary_data(stock_code):
	df = historical_data(stock_code)
	result_dict={}
	if len(df['Close'])<30:

		result_dict["macd"] = -1
		result_dict["emacd"] = -1
		result_dict["rsi"] = -1
		result_dict["stoch_k"] = -1
		result_dict["mfi"] = -1

		result_dict["present_price"] = df['Close'].values[-1]
		result_dict["present_volume"] = df['Volume'].values[-1]
		return result_dict

	macd_series = summary.macd(df['Close'], 12, 26)
	emacd_series = summary.emacd(df['Close'], 12, 26)
	rsi_series = summary.rsi(df['Close'], 14)
	stoch_k_series = summary.stochastic_index(df['Close'], 14)
	mfi_series = summary.mfi(df['Close'], df['Volume'], 14)

	result_dict["macd"] = macd_series.values[-1]
	result_dict["emacd"] = emacd_series.values[-1]
	result_dict["rsi"] = rsi_series.values[-1]
	result_dict["stoch_k"] = stoch_k_series.values[-1]
	result_dict["mfi"] = mfi_series.values[-1]

	result_dict["present_price"] = df['Close'].values[-1]
	result_dict["present_volume"] = df['Volume'].values[-1]

	fin_info_str = finance_info(stock_code=stock_code, chart_type="year")

	return result_dict, fin_info_str

