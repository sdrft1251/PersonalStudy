import FinanceDataReader as fdr
import numpy as np
import pandas as pd

def return_data(code, start_date, end_date):
    return fdr.DataReader(code, start_date, end_date)

def get_stock_code_list(market_code):
    return fdr.StockListing(market_code)

