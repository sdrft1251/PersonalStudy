import FinanceDataReader as fdr
from sklearn.linear_model import LinearRegression
import numpy as np

# 코드에 따른 데이터 불러오기
def return_data(code, start_date, end_date):
    return fdr.DataReader(code, start_date, end_date)

# 시장에 해당하는 모든 코드 불러오기
def get_stock_code_list(market_code):
    return fdr.StockListing(market_code)

# 숫자로만 된 코드 불러오기
def get_code_list_with_digit(code_list):
    new_code_list = []
    for code in code_list:
        if code.isdigit():
            new_code_list.append(code)
    return new_code_list

# 회귀 계수 반환
def return_coef_from_closearr(arr):
    normed_arr = arr / arr[0]
    line_fitter = LinearRegression()
    line_fitter.fit(np.arange(len(normed_arr)).reshape(-1,1), normed_arr)
    return line_fitter.coef_[0]

# Numpy array moving average
def moving_average_for_numpy(arr, window):
    return np.convolve(arr, np.ones(window), 'valid') / window
