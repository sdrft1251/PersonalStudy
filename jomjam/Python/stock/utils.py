import OpenDartReader
import FinanceDataReader as fdr
import datetime
import numpy as np
import pandas as pd

###########################################################################################################################

# OpenDartReader Part

###########################################################################################################################

api_key = 'ebe76fbe2b073384d56fbf6d80e6831615d68acf'


def return_finance_state(code, year):
    """
    inputs
        code -> [String] stock code
        year -> [int] year for searching yyyy
    outputs
        df -> [DataFrame] finance state
    """
    dart = OpenDartReader(api_key)
    df = dart.finstate(code, year)

    return df

def return_stock_num(code, year):
    dart = OpenDartReader(api_key)
    small = dart.report(code, '소액주주', year)
    return int(small.stock_tot_co.values[0].replace(",",""))

def return_stock_info_present(code):
    before_year = datetime.datetime.now().year - 1
    df = fdr.DataReader(code, str(before_year))
    return df.iloc[-1]

def return_finance_info(code, year, target="per"):
    """
    inputs
        code -> [String] stock code
        year -> [Int] year for searching yyyy
        target -> [String] target to know ex) "per"
    outputs
        value -> [Float] target value
    """
    finance_state_df = return_finance_state(code, year)
    present_stock_info = return_stock_info_present(code)
    num_of_stock = return_stock_num(code, year)

    if target == "per":
        net_profit = finance_state_df[np.logical_and(finance_state_df.account_nm=="당기순이익" , finance_state_df.fs_nm=="연결재무제표")].thstrm_amount.values[0]
        eps = int(net_profit.replace(",","")) / num_of_stock
        per = present_stock_info.Close / eps
        return per

    return -1



###########################################################################################################################

# FinanceDataReader Part

###########################################################################################################################



def get_stock_code_list(market_code):
    return fdr.StockListing(market_code)

def find_top_market_cap(stocks_df, year, top=20):
    """
    inputs
        stocks_df => [DataFrame] stock code list for searching from "utils.get_stock_code_list()"
        year =>  [INT] year info. ex) 2021
        top => [INT] num of rank for view (Default = 20)

    outpus
        [LIST] => list of stock code
    """

    # Make date info for input
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-30"
    
    outs = []

    for raw_idx in range(stocks_df.shape[0]):
        code = stocks_df.iloc[raw_idx].Symbol
        name = stocks_df.iloc[raw_idx].Name
        try:
            df = fdr.DataReader(code, start_date, end_date)
            
        except:
            continue
        if 'Close' not in df.columns:
            continue
        elif 'Volume' not in df.columns:
            continue
        close_arr = df.Close.values
        volume_arr = df.Volume.values

        market_cap_arr = close_arr * volume_arr
        market_cap_mean_of_year = np.nanmean(market_cap_arr)
        
        outs.append((name, market_cap_mean_of_year))
    outs.sort(key=lambda x : x[1])
    print(outs[-top:])
    return outs[-top:]
