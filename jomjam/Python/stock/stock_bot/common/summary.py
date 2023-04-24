import pandas as pd
import numpy as np

def ma(series_data, terms):
    return series_data.rolling(window=terms).mean()

def macd(series_data, short_terms, long_terms):
    ma_short = ma(series_data, short_terms)
    ma_long = ma(series_data, long_terms)
    return ma_short - ma_long
    
def hist(series_data, signal_terms):
    signal_series = ma(series_data, signal_terms)
    return series_data - signal_series

def cal_ema(amount_arr, terms):
    tmp_ema=0
    k_1 = 2/(terms+1)
    k_2 = 1-k_1
    for i in range(terms):
        if (i==0):
            tmp_ema = amount_arr[i]
        else:
            tmp_ema = (amount_arr[i]*k_1) + (tmp_ema*k_2)
    return tmp_ema

def ema(series_data, terms):
    ema_list=[]
    for idx in range(len(series_data.values)):
        if (idx<terms-1):
            ema_list.append(series_data.values[idx])
        else:
            ema_list.append(cal_ema(series_data.values[idx-terms+1:idx+1], terms=terms))
    return pd.Series(ema_list, index=series_data.index.values)

def emacd(series_data, short_terms, long_terms):
    ema_short = ema(series_data, short_terms)
    ema_long = ema(series_data, long_terms)
    return ema_short - ema_long

def rsi(series_data, terms):
    shift_series = series_data.shift(periods=1)
    increase_amount = series_data - shift_series
    just_increase = pd.Series(np.where(increase_amount.values>=0, increase_amount.values, 0), index=increase_amount.index.values)
    just_decrease = pd.Series(np.where(increase_amount.values<0, abs(increase_amount.values), 0), index=increase_amount.index.values)
    au = just_increase.rolling(window=terms).mean().fillna(0)
    ad = just_decrease.rolling(window=terms).mean().fillna(0)
    rsi = au/(au+ad+1e-10)
    return rsi

def stochastic_index(series_data, terms):
    l14 = series_data.rolling(window=terms).min().fillna(0)
    h14 = series_data.rolling(window=terms).max().fillna(0)
    stoch_k = (series_data - l14) / (h14 - l14 + 1e-10)
    return stoch_k

def mfi(series_close, series_volume, terms):
    money_flow = series_close * series_volume
    before_price = series_close.shift(periods=1)
    increase_amount = series_close - before_price

    increase_amount_arr = increase_amount.values
    money_flow_arr = money_flow.values

    positive_money_flow_arr = np.where(increase_amount_arr>=0, money_flow_arr, 0)
    negative_money_flow_arr = np.where(increase_amount_arr<0, money_flow_arr, 0)

    positive_money_flow = pd.Series(positive_money_flow_arr, index=series_close.index.values)
    negative_money_flow = pd.Series(negative_money_flow_arr, index=series_close.index.values)

    pmf = positive_money_flow.rolling(window=terms).sum().fillna(0)
    nmf = negative_money_flow.rolling(window=terms).sum().fillna(0)

    mr = pmf / (nmf + 1e-10)
    mfi = (mr) / (1+mr)
    return mfi

