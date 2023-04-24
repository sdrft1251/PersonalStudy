import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from utils import config

have_to_cut_num = 49


def return_macd_signal_macdhist_use_ma_cal(df, short_=12, long_=26, signal_win=9): 
    ma_short = df.close.rolling(window=short_).mean().fillna(0)
    ma_long = df.close.rolling(window=long_).mean().fillna(0)

    macd = ma_short - ma_long
    signal_line = macd.rolling(window=signal_win).mean().fillna(0)
    macd_hist = macd - signal_line

    return macd.values, signal_line.values, macd_hist.values

def return_macd_signal_macdhist_use_ema_cal(df, short_=12, long_=26, signal_win=9):
    ema_short=[]
    ema_long=[]
    for idx in range(len(df.close.values)):
        if (idx<short_-1):
            ema_short.append(df.close.values[idx])
        else:
            ema_short.append(cal_ema(df.close.values[idx-short_+1:idx+1], terms=short_))
            
    for idx in range(len(df.close.values)):
        if (idx<long_-1):
            ema_long.append(df.close.values[idx])
        else:
            ema_long.append(cal_ema(df.close.values[idx-long_+1:idx+1], terms=long_))
    ema_short = pd.Series(ema_short, index=df.index.values)
    ema_long = pd.Series(ema_long, index=df.index.values)
    macd = ema_short - ema_long
    signal_line = macd.rolling(window=signal_win).mean().fillna(0)
    macd_hist = macd - signal_line
    return macd.values, signal_line.values, macd_hist.values

def return_macd_signal_macdhist_use_ema_cal_series_version(ser, short_=12, long_=26, signal_win=9):
    ema_short=[]
    ema_long=[]
    for idx in range(len(ser.values)):
        if (idx<short_-1):
            ema_short.append(ser.values[idx])
        else:
            ema_short.append(cal_ema(ser.values[idx-short_+1:idx+1], terms=short_))
            
    for idx in range(len(ser.values)):
        if (idx<long_-1):
            ema_long.append(ser.values[idx])
        else:
            ema_long.append(cal_ema(ser.values[idx-long_+1:idx+1], terms=long_))
    ema_short = pd.Series(ema_short, index=ser.index.values)
    ema_long = pd.Series(ema_long, index=ser.index.values)
    macd = ema_short - ema_long
    signal_line = macd.rolling(window=signal_win).mean().fillna(0)
    macd_hist = macd - signal_line
    return macd.values, signal_line.values, macd_hist.values

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

def return_ema(ser, terms):
    result_ema=[]
    for idx in range(len(ser)):
        if (idx<terms-1):
            result_ema.append(ser.values[idx])
        else:
            result_ema.append(cal_ema(ser.values[idx-terms+1:idx+1], terms=terms))
    result_ema = pd.Series(result_ema, index=ser.index.values)
    return result_ema
          

def return_rsi(df, term=14):
    before_day_close = df.close.shift(periods=1).fillna(0)
    increase_amount =  df.close-before_day_close
    just_increase = pd.Series(np.where(increase_amount.values>=0, increase_amount.values, 0), index=increase_amount.index.values)
    just_decrease = pd.Series(np.where(increase_amount.values<0, abs(increase_amount.values), 0), index=increase_amount.index.values)
    au = just_increase.rolling(window=term).mean().fillna(0).values
    ad = just_decrease.rolling(window=term).mean().fillna(0).values
    rsi = au/(au+ad+1e-10)
    return rsi

def return_stochastic_index(df, terms=14):
    l14 = df.close.rolling(window=terms).min().fillna(0).values
    h14 = df.close.rolling(window=terms).max().fillna(0).values
    stoch_k = (df.close.values - l14) / (h14 - l14 + 1e-10)
    stoch_k = pd.Series(stoch_k, index=df.index.values)
    stoch_d = stoch_k.rolling(window=3).mean().fillna(0)
    return stoch_k.values, stoch_d.values

def bol_band(df, terms=20):
    tp_val = (df['high'].values + df['low'].values + df['close'].values)/3
    tp_val = pd.Series(tp_val, index=df.index.values)
    ma = tp_val.rolling(window=terms).mean().fillna(0).values
    std = tp_val.rolling(window=terms).std().fillna(0).values
    bolu = ma + 2*std
    bold = ma - 2*std
    return bolu, bold

def return_volume_incline(df, terms=20):
    volume_trend_beta = []
    for idx in range(len(df.volume.values)):
        if (idx<terms-1):
            volume_trend_beta.append(0)
        else:
            volume_trend_beta.append(cal_volume_trend(df.volume.values[idx-terms+1:idx+1]))
    volume_trend_beta = np.array(volume_trend_beta)
    return volume_trend_beta

def return_before_std(df, terms=14):
    before_std = df.close.rolling(window=terms).std().fillna(0).values
    return before_std

def cal_volume_trend(arr):
    min_val = arr.min()
    max_val = arr.max()
    arr = (arr-min_val) / (max_val - min_val + 1e-10)
    lin_model = LinearRegression()
    lin_model.fit(np.arange(len(arr)).reshape(-1,1), arr)
    return lin_model.coef_[0]


def return_trix(df, terms=14, signal_terms=9):
    ema_1 = return_ema(ser=df.close, terms=terms)
    ema_2 = return_ema(ser=ema_1, terms=terms)
    ema_3 = return_ema(ser=ema_2, terms=terms)
    before_ema_3 = ema_3.shift(periods=1).fillna(0)
    trix = ( ema_3.values - before_ema_3.values ) / (before_ema_3.values + 1e-10)
    trix = pd.Series(trix, index=df.index.values)
    trix_signal = trix.rolling(window=signal_terms).mean().fillna(0)
    trix_hist = trix - trix_signal
    return trix.values, trix_signal.values, trix_hist.values


def return_sonar(df, terms=14, signal_terms=9, compare_day_before=1):
    before_close = df.close.shift(periods=compare_day_before).fillna(0)
    today_close_ma = df.close.rolling(window=terms).mean().fillna(0)
    before_close_ma = before_close.rolling(window=terms).mean().fillna(0)

    sonar = ( today_close_ma.values - before_close_ma.values ) / (before_close_ma.values + 1e-10)
    sonar = pd.Series(sonar, index=df.index.values)
    signar = sonar.rolling(window=signal_terms).mean().fillna(0)
    sonar_hist = sonar - signar
    return sonar.values, signar.values, sonar_hist.values


def return_sigma(df, terms=20, signal_terms=9):
    today_ma = df.close.rolling(window=terms).mean().fillna(0)
    today_std = df.close.rolling(window=terms).std().fillna(0)

    sigma = ( df.close.values - today_ma.values ) / (today_std.values + 1e-10)
    sigma = pd.Series(sigma, index=df.index.values)
    sigma_signal = sigma.rolling(window=signal_terms).mean().fillna(0)
    sigma_hist = sigma - sigma_signal
    return sigma.values, sigma_signal.values, sigma_hist.values

def return_mfi(df, terms=14, signal_terms=9):
    standard_price = (df.close.values + df.high.values + df.low.values) / 3
    standard_price = pd.Series(standard_price, index=df.index.values)
    money_flow = (standard_price.values * df.volume.values)
    money_flow = pd.Series(money_flow, index=df.index.values)
    before_price = df.close.shift(periods=1).fillna(0)
    increase_amount = df.close - before_price

    increase_amount_arr = increase_amount.values
    money_flow_arr = money_flow.values

    positive_money_flow_arr = np.where(increase_amount_arr>=0, money_flow_arr, 0)
    negative_money_flow_arr = np.where(increase_amount_arr<0, money_flow_arr, 0)

    positive_money_flow = pd.Series(positive_money_flow_arr, index=df.index.values)
    negative_money_flow = pd.Series(negative_money_flow_arr, index=df.index.values)

    pmf = positive_money_flow.rolling(window=terms).sum().fillna(0)
    nmf = negative_money_flow.rolling(window=terms).sum().fillna(0)

    mr = pmf.values / (nmf.values + 1e-10)
    mfi = (mr) / (1+mr+1e-10)
    mfi = pd.Series(mfi, index=df.index.values)
    mfi_signal = mfi.rolling(window=signal_terms).mean().fillna(0)
    mfi_hist = mfi - mfi_signal
    return mfi.values, mfi_signal.values, mfi_hist.values

def return_mfi_50(df, terms=14, before_temrs=1):
    standard_price = (df.close.values + df.high.values + df.low.values) / 3
    standard_price = pd.Series(standard_price, index=df.index.values)
    money_flow = (standard_price.values * df.volume.values)
    money_flow = pd.Series(money_flow, index=df.index.values)
    before_price = df.close.shift(periods=1).fillna(0)
    increase_amount = df.close - before_price

    increase_amount_arr = increase_amount.values
    money_flow_arr = money_flow.values

    positive_money_flow_arr = np.where(increase_amount_arr>=0, money_flow_arr, 0)
    negative_money_flow_arr = np.where(increase_amount_arr<0, money_flow_arr, 0)

    positive_money_flow = pd.Series(positive_money_flow_arr, index=df.index.values)
    negative_money_flow = pd.Series(negative_money_flow_arr, index=df.index.values)

    pmf = positive_money_flow.rolling(window=terms).sum().fillna(0)
    nmf = negative_money_flow.rolling(window=terms).sum().fillna(0)

    mr = pmf.values / (nmf.values + 1e-10)
    mfi = (mr) / (1+mr+1e-10)
    mfi_50 = mfi-0.5

    mfi_50 = pd.Series(mfi_50, index=df.index.values)
    mfi_50_pass_zero = return_pass_zero(ser=mfi_50, before_terms = before_temrs)
    mfi_50_pass_zero_tw = return_pass_zero_towhere(ser=mfi_50, before_terms = before_temrs)
    return mfi_50_pass_zero, mfi_50_pass_zero_tw

def return_tanh_inclination(ser, before_terms=1):
    ser = pd.Series(ser)
    before_ser = ser.shift(periods=before_terms).values[before_terms:].reshape(-1,1)
    today_ser = ser.values[before_terms:].reshape(-1,1)

    robust_scaler = RobustScaler()
    robust_scaler.fit(today_ser)

    today_ser = robust_scaler.transform(today_ser)
    before_ser = robust_scaler.transform(before_ser)

    today_ser = today_ser.reshape(-1)
    before_ser = before_ser.reshape(-1)

    ser_tanh = np.tanh(today_ser)
    before_ser_tanh = np.tanh(before_ser)

    ser_tanh_inclination = (ser_tanh - before_ser_tanh) / (today_ser - before_ser + 1e-10)
    tmp_zeros = np.zeros(before_terms)
    ser_tanh_inclination = np.concatenate((tmp_zeros,ser_tanh_inclination), axis=0)
    return ser_tanh_inclination

def return_exp_inclination(ser, before_terms=1):
    ser = pd.Series(ser)
    before_ser = ser.shift(periods=before_terms).values[before_terms:].reshape(-1,1)
    today_ser = ser.values[before_terms:].reshape(-1,1)

    robust_scaler = RobustScaler()
    robust_scaler.fit(today_ser)

    today_ser = robust_scaler.transform(today_ser)
    before_ser = robust_scaler.transform(before_ser)

    today_ser = today_ser.reshape(-1)
    before_ser = before_ser.reshape(-1)

    ser_exp = (today_ser*today_ser*today_ser) + today_ser
    before_ser_exp = (before_ser*before_ser*before_ser) + before_ser

    ser_exp_inclination = (ser_exp - before_ser_exp) / (today_ser - before_ser + 1e-10)
    tmp_zeros = np.zeros(before_terms)
    ser_exp_inclination = np.concatenate((tmp_zeros,ser_exp_inclination), axis=0)
    return ser_exp_inclination

def return_sin_inclination(ser, before_terms=1):
    ser = pd.Series(ser)
    before_ser = ser.shift(periods=before_terms).values[before_terms:].reshape(-1,1)
    today_ser = ser.values[before_terms:].reshape(-1,1)

    robust_scaler = RobustScaler()
    robust_scaler.fit(today_ser)

    today_ser = robust_scaler.transform(today_ser)
    before_ser = robust_scaler.transform(before_ser)

    today_ser = today_ser.reshape(-1)
    before_ser = before_ser.reshape(-1)

    ser_sin = np.sin(today_ser)
    before_ser_sin = np.sin(before_ser)

    ser_sin_inclination = (ser_sin - before_ser_sin) / (today_ser - before_ser + 1e-10)
    tmp_zeros = np.zeros(before_terms)
    ser_sin_inclination = np.concatenate((tmp_zeros,ser_sin_inclination), axis=0)
    return ser_sin_inclination

def return_pass_zero(ser, before_terms=1):
    ser = pd.Series(ser)
    beofre_ser = ser.shift(periods=before_terms).values
    today_ser = ser.values
    mul_today_before = today_ser * beofre_ser
    return mul_today_before
    
def return_pass_zero_towhere(ser, before_terms=1):
    ser = pd.Series(ser)
    beofre_ser = ser.shift(periods=before_terms).values
    today_ser = ser.values
    mul_today_before = today_ser * beofre_ser
    mul_today_before_only_pass_zero = np.where(mul_today_before<0, -mul_today_before, 0)
    tmp_for_disting = np.where(today_ser>=0, 1, -1)
    pass_zero_tw = mul_today_before_only_pass_zero * tmp_for_disting
    return pass_zero_tw

def return_momentum(ser, before_temrs=1):
    ser = pd.Series(ser)
    stand_arr = ser.values
    before_arr = ser.shift(periods=before_temrs).fillna(0).values
    momentum = stand_arr - before_arr
    return momentum


def return_linear_regression_coeff_and_r_square(arr):
    reg = LinearRegression().fit(np.arange(len(arr)).reshape(-1,1), arr)
    coef_val = reg.coef_[0]
    r_square_val = reg.score(np.arange(len(arr)).reshape(-1,1), arr)
    return coef_val, r_square_val

def return_lrs(df, terms=14, signal_terms=9):
    lrs_list = []
    r_square_list=[]
    for idx in range(len(df.close.values)):
        if (idx<terms-1):
            lrs_list.append(0)
            r_square_list.append(0)
        else:
            lr, sq = return_linear_regression_coeff_and_r_square(df.close.values[idx-terms+1:idx+1])
            lrs_list.append(lr)
            r_square_list.append(sq)
    
    lrs = pd.Series(lrs_list, index=df.index.values)
    r_square = pd.Series(r_square_list, index=df.index.values)
    
    lrs_signal = lrs.rolling(window=signal_terms).mean().fillna(0)
    lrs_hist = lrs - lrs_signal

    r_square_signal = r_square.rolling(window=signal_terms).mean().fillna(0)
    r_square_hist = r_square - r_square_signal

    before_lrs = lrs.shift(periods=1).fillna(0)
    lrs_rate_of_change = (lrs.values - before_lrs.values) / (before_lrs.values + 1e-10)
    lrs_rate_of_change = pd.Series(lrs_rate_of_change, index=df.index.values)
    lrs_rate_of_change_signal = lrs_rate_of_change.rolling(window=signal_terms).mean().fillna(0)
    lrs_rate_of_change_hist = lrs_rate_of_change - lrs_rate_of_change_signal

    return lrs.values, lrs_signal.values, lrs_hist.values, lrs_rate_of_change.values, lrs_rate_of_change_signal.values, lrs_rate_of_change_hist.values, r_square.values, r_square_signal.values, r_square_hist.values


def return_atr(df, signal_terms=9):
    today_high = df.high
    today_low = df.low
    before_close = df.close.shift(periods=1).fillna(0)
    
    tr_1 = today_high - today_low
    tr_2 = today_high - before_close
    tr_3 = today_low - before_close

    atr = (tr_1.values + tr_2.values + tr_3.values)/3
    atr = pd.Series(atr, index=df.index.values)
    before_atr = atr.shift(periods=1).fillna(0)

    atr_change_rate = (atr.values-before_atr.values) / (before_atr.values + 1e-10)
    atr_change_rate = pd.Series(atr_change_rate, index=df.index.values)

    atr_change_rate_signal = atr_change_rate.rolling(window=signal_terms).mean().fillna(0)
    atr_change_rate_hist = atr_change_rate - atr_change_rate_signal

    return atr_change_rate.values, atr_change_rate_signal.values, atr_change_rate_hist.values

def return_dema_tema(df, ema_terms=9):
    ema_se = return_ema(ser=df.close, terms=ema_terms)
    dema_se = return_ema(ser=ema_se, terms=ema_terms)
    tema_se = return_ema(ser=dema_se, terms=ema_terms)
    return dema_se.values, tema_se.values


def return_volume_increase_rate(df, before_terms=1):
    before_volume = df.volume.shift(periods=before_terms).fillna(0)
    volume_change_rate = (df.volume.values) / (before_volume.values + 1e-10)
    return volume_change_rate

def return_increase_rate(ser, before_terms=1):
    if (str(type(ser)) == "<class 'numpy.ndarray'>"):
        ser = pd.Series(ser)
    before_ser = ser.shift(periods=before_terms).fillna(0)
    ser_change_rate = (ser.values-before_ser.values) / (before_ser.values+1e-10)
    return ser_change_rate


def return_cci_change_rate(df, terms=20, before_terms=1):
    m_price = (df.close.values + df.high.values + df.low.values)/3
    n_price = df.close.rolling(window=terms).mean().fillna(0).values
    d_price = (m_price-n_price)/terms
    cci = (m_price * n_price) / (0.015*d_price)
    cci = pd.Series(cci)
    cci_change_rate = return_increase_rate(ser=cci, before_terms=before_terms)
    return cci_change_rate


def return_adx(df, before_terms=1):
    high = df.high.values
    before_high = df.high.shift(periods=before_terms).values
    low = df.low.values
    before_low = df.low.shift(periods=before_terms).values

    high_increase = high - before_high
    low_increase = low - before_low

    pdi = np.where(high_increase>=0, high_increase, 0)
    mdi = np.where(low_increase>=0, -low_increase, 0)

    pdi_mdi_diff = pdi - mdi
    adx = abs(pdi - mdi) / (pdi + mdi)
    return pdi_mdi_diff, adx
