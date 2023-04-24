import pandas
from utils import tech_indicator, support
import numpy as np

basic_input_col_name_list = [
        'macd_ma', 'signal_line_ma', 'macd_hist_ma', 'macd_ema', 'signal_line_ema', 'macd_hist_ema', 'rsi', 'stoch_k', 'stoch_d', 'stoch_hist', 'bol_rate', 'volume_trend_incline',\
        'trix', 'trix_signal', 'trix_hist', 'sonar', 'signar', 'sonar_hist',\
        'sigma', 'sigma_signal', 'sigma_hist', 'mfi', 'mfi_signal', 'mfi_hist', 'mfi_50_pass_zero', 'mfi_50_pass_zero_tw',\
        'lrs', 'lrs_signal', 'lrs_hist', 'lrs_rate_of_change', 'lrs_rate_of_change_signal', 'lrs_rate_of_change_hist', 'r_square', 'r_square_signal', 'r_square_hist',\
        'atr_change_rate', 'atr_change_rate_signal', 'atr_change_rate_hist', 'dema_se', 'tema_se', 'close_change_rate', 'volume_change_rate',\
        'before_std', 'cci_change_rate', 'pdi_mdi_diff', 'adx', 'adx_increase_rate', 'adx_pass_zero', 'adx_pass_zero_tw'\
        ]

feature_enginnering_input_col_name_list =[\
        'macd_ma_momentum', 'macd_ma_abs', 'macd_ma_tanh_inclination', 'macd_ma_exp_inclination', 'macd_ma_sin_inclination', 'macd_ma_pass_zero', 'macd_ma_pass_zero_tw', 'signal_ma_momentum', \
        'macd_hist_ma_momentum', 'macd_hist_ma_abs', 'macd_hist_ma_tanh_inclination', 'macd_hist_ma_exp_inclination', 'macd_hist_ma_sin_inclination', 'macd_hist_ma_pass_zero', 'macd_hist_ma_pass_zero_tw',\
        'macd_ema_momentum', 'macd_ema_abs', 'macd_ema_tanh_inclination', 'macd_ema_exp_inclination', 'macd_ema_sin_inclination', 'macd_ema_pass_zero', 'macd_ema_pass_zero_tw', 'signal_ema_momentum',\
        'macd_hist_ema_momentum', 'macd_hist_ema_abs', 'macd_hist_ema_tanh_inclination', 'macd_hist_ema_exp_inclination', 'macd_hist_ema_sin_inclination', 'macd_hist_ema_pass_zero', 'macd_hist_ema_pass_zero_tw',\
        'stoch_hist_momentum', 'stoch_hist_abs', 'stoch_hist_tanh_inclination', 'stoch_hist_exp_inclination', 'stoch_hist_sin_inclination', 'stoch_hist_pass_zero', 'stoch_hist_pass_zero_tw',\
        'trix_hist_momentum', 'trix_hist_abs', 'trix_hist_tanh_inclination', 'trix_hist_exp_inclination', 'trix_hist_sin_inclination', 'trix_hist_pass_zero', 'trix_hist_pass_zero_tw',\
        'sonar_pass_zero', 'sonar_pass_zero_tw',\
        'sonar_hist_momentum', 'sonar_hist_abs', 'sonar_hist_tanh_inclination', 'sonar_hist_exp_inclination', 'sonar_hist_sin_inclination', 'sonar_hist_pass_zero', 'sonar_hist_pass_zero_tw',\
        'sigma_hist_momentum', 'sigma_hist_abs', 'sigma_hist_tanh_inclination', 'sigma_hist_exp_inclination', 'sigma_hist_sin_inclination', 'sigma_hist_pass_zero', 'sigma_hist_pass_zero_tw',\
        'mfi_momentum', 'mfi_tanh_inclination', 'mfi_exp_inclination', 'mfi_sin_inclination',\
        'mfi_hist_momentum', 'mfi_hist_abs', 'mfi_hist_tanh_inclination', 'mfi_hist_exp_inclination', 'mfi_hist_sin_inclination', 'mfi_hist_pass_zero', 'mfi_hist_pass_zero_tw',\
        'lrs_momentum', 'lrs_abs', 'lrs_tanh_inclination', 'lrs_exp_inclination', 'lrs_sin_inclination',\
        'lrs_hist_momentum', 'lrs_hist_abs', 'lrs_hist_tanh_inclination', 'lrs_hist_exp_inclination', 'lrs_hist_sin_inclination', 'lrs_hist_pass_zero', 'lrs_hist_pass_zero_tw',\
        'lrs_rate_of_change_momentum', 'lrs_rate_of_change_abs', 'lrs_rate_of_change_tanh_inclination', 'lrs_rate_of_change_exp_inclination', 'lrs_rate_of_change_sin_inclination',\
        'lrs_rate_of_change_hist_momentum', 'lrs_rate_of_change_hist_abs', 'lrs_rate_of_change_hist_tanh_inclination', 'lrs_rate_of_change_hist_exp_inclination', 'lrs_rate_of_change_hist_sin_inclination', 'lrs_rate_of_change_hist_pass_zero', 'lrs_rate_of_change_hist_pass_zero_tw',\
        'r_square_momentum', 'r_square_tanh_inclination', 'r_square_exp_inclination', 'r_square_sin_inclination',\
        'r_square_hist_momentum', 'r_square_hist_abs', 'r_square_hist_tanh_inclination', 'r_square_hist_exp_inclination', 'r_square_hist_sin_inclination', 'r_square_hist_pass_zero', 'r_square_hist_pass_zero_tw',\
        'atr_change_rate_momentum', 'atr_change_rate_abs', 'atr_change_rate_tanh_inclination', 'atr_change_rate_exp_inclination', 'atr_change_rate_sin_inclination',\
        'atr_change_rate_hist_momentum', 'atr_change_rate_hist_abs', 'atr_change_rate_hist_tanh_inclination', 'atr_change_rate_hist_exp_inclination', 'atr_change_rate_hist_sin_inclination', 'atr_change_rate_hist_pass_zero', 'atr_change_rate_hist_pass_zero_tw',\
        'dema_se_momentum', 'dema_se_tanh_inclination', 'dema_se_exp_inclination', 'dema_se_sin_inclination',\
        'tema_se_momentum', 'tema_se_tanh_inclination', 'tema_se_exp_inclination', 'tema_se_sin_inclination',\
        'close_change_rate_momentum', 'close_change_rate_abs', 'close_change_rate_tanh_inclination', 'close_change_rate_exp_inclination', 'close_change_rate_sin_inclination',\
        'volume_change_rate_momentum', 'volume_change_rate_abs', 'volume_change_rate_tanh_inclination', 'volume_change_rate_exp_inclination', 'volume_change_rate_sin_inclination',\
        'box_and_mfi_50_pass_zero', 'box_and_mfi_50_pass_zero_tw', 'cci_X_close_change_rate'\
        ]
nasdaq_input_col_name_list=[
        'macd_nasdaq', 'signal_line_nasdaq', 'macd_hist_nasdaq',\
        'macd_nasdaq_momentum', 'macd_nasdaq_abs', 'macd_nasdaq_tanh_inclination', 'macd_nasdaq_exp_inclination', 'macd_nasdaq_sin_inclination',\
        'macd_hist_nasdaq_momentum', 'macd_hist_nasdaq_abs', 'macd_hist_nasdaq_tanh_inclination', 'macd_hist_nasdaq_exp_inclination', 'macd_hist_nasdaq_sin_inclination',\
        'nasdaq_change_rate',\
        'nasdaq_change_rate_momentum', 'nasdaq_change_rate_abs', 'nasdaq_change_rate_tanh_inclination', 'nasdaq_change_rate_exp_inclination', 'nasdaq_change_rate_sin_inclination'\
    ]

tot_col_names = basic_input_col_name_list + feature_enginnering_input_col_name_list + nasdaq_input_col_name_list

def return_preprocessed_df(df, profit_terms, data_split_type):
    ###### Label Column
    start=True
    label_col_list=[]
    for te in profit_terms:
        day_after_close = df.close.shift(periods=-te).fillna(0)
        tmp_increase_rate = day_after_close.values / (df.close.values + 1e-10)
        if start:
            increase_rate_arr = tmp_increase_rate.reshape(-1,1)
            label_col_list.append("increase_rate_after_"+str(te))
            start=False
        else:
            increase_rate_arr = np.concatenate((increase_rate_arr, tmp_increase_rate.reshape(-1,1)), axis=1)
            label_col_list.append("increase_rate_after_"+str(te))

    ###### Make Feature Values

    #Basic value
    macd_ma, signal_line_ma, macd_hist_ma = tech_indicator.return_macd_signal_macdhist_use_ma_cal(df=df, short_=12, long_=26, signal_win=9)
    macd_ema, signal_line_ema, macd_hist_ema = tech_indicator.return_macd_signal_macdhist_use_ema_cal(df=df, short_=12, long_=26, signal_win=9)

    macd_ma = (macd_ma) / (df.close.values + 1e-10)
    signal_line_ma = (signal_line_ma) / (df.close.values + 1e-10)
    macd_hist_ma = (macd_hist_ma) / (df.close.values + 1e-10)
    macd_ema = (macd_ema) / (df.close.values + 1e-10)
    signal_line_ema = (signal_line_ema) / (df.close.values + 1e-10)
    macd_hist_ema = (macd_hist_ema) / (df.close.values + 1e-10)

    rsi = tech_indicator.return_rsi(df=df, term=14)
    stoch_k, stoch_d = tech_indicator.return_stochastic_index(df=df, terms=14)
    stoch_hist = stoch_k-stoch_d
    bolu, bold = tech_indicator.bol_band(df=df, terms=20)
    bol_rate = (df.close.values-bold) / (bolu-bold+1e-10)

    volume_trend_incline = tech_indicator.return_volume_incline(df=df, terms=20)
    trix, trix_signal, trix_hist = tech_indicator.return_trix(df=df, terms=14, signal_terms=9)

    sonar, signar, sonar_hist = tech_indicator.return_sonar(df=df, terms=14, signal_terms=9, compare_day_before=1)
    sigma, sigma_signal, sigma_hist = tech_indicator.return_sigma(df=df, terms=20, signal_terms=9)

    mfi, mfi_signal, mfi_hist = tech_indicator.return_mfi(df=df, terms=14, signal_terms=9)
    mfi_50_pass_zero, mfi_50_pass_zero_tw = tech_indicator.return_mfi_50(df=df, terms=14, before_temrs=1)
    lrs, lrs_signal, lrs_hist, lrs_rate_of_change, lrs_rate_of_change_signal, lrs_rate_of_change_hist, r_square, r_square_signal, r_square_hist = tech_indicator.return_lrs(df=df, terms=14, signal_terms=9)
    atr_change_rate, atr_change_rate_signal, atr_change_rate_hist = tech_indicator.return_atr(df=df, signal_terms=9)
    dema_se, tema_se = tech_indicator.return_dema_tema(df=df, ema_terms=9)

    dema_se = (dema_se) / (df.close.values + 1e-10)
    tema_se = (tema_se) / (df.close.values + 1e-10)

    close_change_rate = tech_indicator.return_increase_rate(ser=df.close, before_terms=1)
    volume_change_rate = tech_indicator.return_increase_rate(ser=df.volume, before_terms=1)

    before_std = tech_indicator.return_before_std(df=df, terms=14)
    cci_change_rate = tech_indicator.return_cci_change_rate(df=df, terms=20, before_terms=1)
    pdi_mdi_diff, adx = tech_indicator.return_adx(df=df, before_terms=1)
    adx_increase_rate = tech_indicator.return_increase_rate(ser=adx, before_terms=1)
    adx_pass_zero = tech_indicator.return_pass_zero(ser=adx, before_terms=1)
    adx_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=adx, before_terms=1)

    basic_feature_arrs = [
        macd_ma, signal_line_ma, macd_hist_ma, macd_ema, signal_line_ema, macd_hist_ema, rsi, stoch_k, stoch_d, stoch_hist, bol_rate, volume_trend_incline,\
        trix, trix_signal, trix_hist, sonar, signar, sonar_hist,\
        sigma, sigma_signal, sigma_hist, mfi, mfi_signal, mfi_hist, mfi_50_pass_zero, mfi_50_pass_zero_tw,\
        lrs, lrs_signal, lrs_hist, lrs_rate_of_change, lrs_rate_of_change_signal, lrs_rate_of_change_hist, r_square, r_square_signal, r_square_hist,\
        atr_change_rate, atr_change_rate_signal, atr_change_rate_hist, dema_se, tema_se, close_change_rate, volume_change_rate,\
        before_std, cci_change_rate, pdi_mdi_diff, adx, adx_increase_rate, adx_pass_zero, adx_pass_zero_tw\
    ]

    ########################### Feature Engineer #####################################
    macd_ma_momentum = tech_indicator.return_momentum(ser=macd_ma, before_temrs=1)
    macd_ma_abs = abs(macd_ma)
    macd_ma_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_ma, before_terms=1)
    macd_ma_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_ma, before_terms=1)
    macd_ma_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_ma, before_terms=1)
    macd_ma_pass_zero = tech_indicator.return_pass_zero(ser=macd_ma, before_terms=1)
    macd_ma_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=macd_ma, before_terms=1)

    signal_ma_momentum = tech_indicator.return_momentum(ser=signal_line_ma, before_temrs=1)

    macd_hist_ma_momentum = tech_indicator.return_momentum(ser=macd_hist_ma, before_temrs=1)
    macd_hist_ma_abs = abs(macd_hist_ma)
    macd_hist_ma_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_hist_ma, before_terms=1)
    macd_hist_ma_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_hist_ma, before_terms=1)
    macd_hist_ma_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_hist_ma, before_terms=1)
    macd_hist_ma_pass_zero = tech_indicator.return_pass_zero(ser=macd_hist_ma, before_terms=1)
    macd_hist_ma_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=macd_hist_ma, before_terms=1)

    macd_ema_momentum = tech_indicator.return_momentum(ser=macd_ema, before_temrs=1)
    macd_ema_abs = abs(macd_ema)
    macd_ema_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_ema, before_terms=1)
    macd_ema_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_ema, before_terms=1)
    macd_ema_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_ema, before_terms=1)
    macd_ema_pass_zero = tech_indicator.return_pass_zero(ser=macd_ema, before_terms=1)
    macd_ema_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=macd_ema, before_terms=1)

    signal_ema_momentum = tech_indicator.return_momentum(ser=signal_line_ema, before_temrs=1)

    macd_hist_ema_momentum = tech_indicator.return_momentum(ser=macd_hist_ema, before_temrs=1)
    macd_hist_ema_abs = abs(macd_hist_ema)
    macd_hist_ema_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_hist_ema, before_terms=1)
    macd_hist_ema_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_hist_ema, before_terms=1)
    macd_hist_ema_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_hist_ema, before_terms=1)
    macd_hist_ema_pass_zero = tech_indicator.return_pass_zero(ser=macd_hist_ema, before_terms=1)
    macd_hist_ema_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=macd_hist_ema, before_terms=1)

    stoch_hist_momentum = tech_indicator.return_momentum(ser=stoch_hist, before_temrs=1)
    stoch_hist_abs = abs(stoch_hist)
    stoch_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=stoch_hist, before_terms=1)
    stoch_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=stoch_hist, before_terms=1)
    stoch_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=stoch_hist, before_terms=1)
    stoch_hist_pass_zero = tech_indicator.return_pass_zero(ser=stoch_hist, before_terms=1)
    stoch_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=stoch_hist, before_terms=1)

    trix_hist_momentum = tech_indicator.return_momentum(ser=trix_hist, before_temrs=1)
    trix_hist_abs = abs(trix_hist)
    trix_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=trix_hist, before_terms=1)
    trix_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=trix_hist, before_terms=1)
    trix_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=trix_hist, before_terms=1)
    trix_hist_pass_zero = tech_indicator.return_pass_zero(ser=trix_hist, before_terms=1)
    trix_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=trix_hist, before_terms=1)

    sonar_pass_zero = tech_indicator.return_pass_zero(ser=sonar, before_terms=1)
    sonar_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=sonar, before_terms=1)

    sonar_hist_momentum = tech_indicator.return_momentum(ser=sonar_hist, before_temrs=1)
    sonar_hist_abs = abs(sonar_hist)
    sonar_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=sonar_hist, before_terms=1)
    sonar_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=sonar_hist, before_terms=1)
    sonar_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=sonar_hist, before_terms=1)
    sonar_hist_pass_zero = tech_indicator.return_pass_zero(ser=sonar_hist, before_terms=1)
    sonar_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=sonar_hist, before_terms=1)

    sigma_hist_momentum = tech_indicator.return_momentum(ser=sigma_hist, before_temrs=1)
    sigma_hist_abs = abs(sigma_hist)
    sigma_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=sigma_hist, before_terms=1)
    sigma_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=sigma_hist, before_terms=1)
    sigma_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=sigma_hist, before_terms=1)
    sigma_hist_pass_zero = tech_indicator.return_pass_zero(ser=sigma_hist, before_terms=1)
    sigma_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=sigma_hist, before_terms=1)

    mfi_momentum = tech_indicator.return_momentum(ser=mfi, before_temrs=1)
    mfi_tanh_inclination = tech_indicator.return_tanh_inclination(ser=mfi, before_terms=1)
    mfi_exp_inclination = tech_indicator.return_exp_inclination(ser=mfi, before_terms=1)
    mfi_sin_inclination = tech_indicator.return_sin_inclination(ser=mfi, before_terms=1)

    mfi_hist_momentum = tech_indicator.return_momentum(ser=mfi_hist, before_temrs=1)
    mfi_hist_abs = abs(mfi_hist)
    mfi_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=mfi_hist, before_terms=1)
    mfi_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=mfi_hist, before_terms=1)
    mfi_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=mfi_hist, before_terms=1)
    mfi_hist_pass_zero = tech_indicator.return_pass_zero(ser=mfi_hist, before_terms=1)
    mfi_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=mfi_hist, before_terms=1)

    lrs_momentum = tech_indicator.return_momentum(ser=lrs, before_temrs=1)
    lrs_abs = abs(lrs)
    lrs_tanh_inclination = tech_indicator.return_tanh_inclination(ser=lrs, before_terms=1)
    lrs_exp_inclination = tech_indicator.return_exp_inclination(ser=lrs, before_terms=1)
    lrs_sin_inclination = tech_indicator.return_sin_inclination(ser=lrs, before_terms=1)

    lrs_hist_momentum = tech_indicator.return_momentum(ser=lrs_hist, before_temrs=1)
    lrs_hist_abs = abs(lrs_hist)
    lrs_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=lrs_hist, before_terms=1)
    lrs_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=lrs_hist, before_terms=1)
    lrs_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=lrs_hist, before_terms=1)
    lrs_hist_pass_zero = tech_indicator.return_pass_zero(ser=lrs_hist, before_terms=1)
    lrs_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=lrs_hist, before_terms=1)

    lrs_rate_of_change_momentum = tech_indicator.return_momentum(ser=lrs_rate_of_change, before_temrs=1)
    lrs_rate_of_change_abs = abs(lrs_rate_of_change)
    lrs_rate_of_change_tanh_inclination = tech_indicator.return_tanh_inclination(ser=lrs_rate_of_change, before_terms=1)
    lrs_rate_of_change_exp_inclination = tech_indicator.return_exp_inclination(ser=lrs_rate_of_change, before_terms=1)
    lrs_rate_of_change_sin_inclination = tech_indicator.return_sin_inclination(ser=lrs_rate_of_change, before_terms=1)

    lrs_rate_of_change_hist_momentum = tech_indicator.return_momentum(ser=lrs_rate_of_change_hist, before_temrs=1)
    lrs_rate_of_change_hist_abs = abs(lrs_rate_of_change_hist)
    lrs_rate_of_change_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=lrs_rate_of_change_hist, before_terms=1)
    lrs_rate_of_change_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=lrs_rate_of_change_hist, before_terms=1)
    lrs_rate_of_change_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=lrs_rate_of_change_hist, before_terms=1)
    lrs_rate_of_change_hist_pass_zero = tech_indicator.return_pass_zero(ser=lrs_rate_of_change_hist, before_terms=1)
    lrs_rate_of_change_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=lrs_rate_of_change_hist, before_terms=1)

    r_square_momentum = tech_indicator.return_momentum(ser=r_square, before_temrs=1)
    r_square_tanh_inclination = tech_indicator.return_tanh_inclination(ser=r_square, before_terms=1)
    r_square_exp_inclination = tech_indicator.return_exp_inclination(ser=r_square, before_terms=1)
    r_square_sin_inclination = tech_indicator.return_sin_inclination(ser=r_square, before_terms=1)

    r_square_hist_momentum = tech_indicator.return_momentum(ser=r_square_hist, before_temrs=1)
    r_square_hist_abs = abs(r_square_hist)
    r_square_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=r_square_hist, before_terms=1)
    r_square_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=r_square_hist, before_terms=1)
    r_square_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=r_square_hist, before_terms=1)
    r_square_hist_pass_zero = tech_indicator.return_pass_zero(ser=r_square_hist, before_terms=1)
    r_square_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=r_square_hist, before_terms=1)

    atr_change_rate_momentum = tech_indicator.return_momentum(ser=atr_change_rate, before_temrs=1)
    atr_change_rate_abs = abs(atr_change_rate)
    atr_change_rate_tanh_inclination = tech_indicator.return_tanh_inclination(ser=atr_change_rate, before_terms=1)
    atr_change_rate_exp_inclination = tech_indicator.return_exp_inclination(ser=atr_change_rate, before_terms=1)
    atr_change_rate_sin_inclination = tech_indicator.return_sin_inclination(ser=atr_change_rate, before_terms=1)

    atr_change_rate_hist_momentum = tech_indicator.return_momentum(ser=atr_change_rate_hist, before_temrs=1)
    atr_change_rate_hist_abs = abs(atr_change_rate_hist)
    atr_change_rate_hist_tanh_inclination = tech_indicator.return_tanh_inclination(ser=atr_change_rate_hist, before_terms=1)
    atr_change_rate_hist_exp_inclination = tech_indicator.return_exp_inclination(ser=atr_change_rate_hist, before_terms=1)
    atr_change_rate_hist_sin_inclination = tech_indicator.return_sin_inclination(ser=atr_change_rate_hist, before_terms=1)
    atr_change_rate_hist_pass_zero = tech_indicator.return_pass_zero(ser=atr_change_rate_hist, before_terms=1)
    atr_change_rate_hist_pass_zero_tw = tech_indicator.return_pass_zero_towhere(ser=atr_change_rate_hist, before_terms=1)

    dema_se_momentum = tech_indicator.return_momentum(ser=dema_se, before_temrs=1)
    dema_se_tanh_inclination = tech_indicator.return_tanh_inclination(ser=dema_se, before_terms=1)
    dema_se_exp_inclination = tech_indicator.return_exp_inclination(ser=dema_se, before_terms=1)
    dema_se_sin_inclination = tech_indicator.return_sin_inclination(ser=dema_se, before_terms=1)

    tema_se_momentum = tech_indicator.return_momentum(ser=tema_se, before_temrs=1)
    tema_se_tanh_inclination = tech_indicator.return_tanh_inclination(ser=tema_se, before_terms=1)
    tema_se_exp_inclination = tech_indicator.return_exp_inclination(ser=tema_se, before_terms=1)
    tema_se_sin_inclination = tech_indicator.return_sin_inclination(ser=tema_se, before_terms=1)

    close_change_rate_momentum = tech_indicator.return_momentum(ser=close_change_rate, before_temrs=1)
    close_change_rate_abs = abs(close_change_rate)
    close_change_rate_tanh_inclination = tech_indicator.return_tanh_inclination(ser=close_change_rate, before_terms=1)
    close_change_rate_exp_inclination = tech_indicator.return_exp_inclination(ser=close_change_rate, before_terms=1)
    close_change_rate_sin_inclination = tech_indicator.return_sin_inclination(ser=close_change_rate, before_terms=1)

    volume_change_rate_momentum = tech_indicator.return_momentum(ser=volume_change_rate, before_temrs=1)
    volume_change_rate_abs = abs(volume_change_rate)
    volume_change_rate_tanh_inclination = tech_indicator.return_tanh_inclination(ser=volume_change_rate, before_terms=1)
    volume_change_rate_exp_inclination = tech_indicator.return_exp_inclination(ser=volume_change_rate, before_terms=1)
    volume_change_rate_sin_inclination = tech_indicator.return_sin_inclination(ser=volume_change_rate, before_terms=1)

    box_and_mfi_50_pass_zero = mfi_50_pass_zero / (before_std + 1e-10)
    box_and_mfi_50_pass_zero_tw = mfi_50_pass_zero_tw / (before_std + 1e-10)
    cci_X_close_change_rate = cci_change_rate * close_change_rate


    feature_enginnered_arrs =[\
        macd_ma_momentum, macd_ma_abs, macd_ma_tanh_inclination, macd_ma_exp_inclination, macd_ma_sin_inclination, macd_ma_pass_zero, macd_ma_pass_zero_tw, signal_ma_momentum,\
        macd_hist_ma_momentum, macd_hist_ma_abs, macd_hist_ma_tanh_inclination, macd_hist_ma_exp_inclination, macd_hist_ma_sin_inclination, macd_hist_ma_pass_zero, macd_hist_ma_pass_zero_tw,\
        macd_ema_momentum, macd_ema_abs, macd_ema_tanh_inclination, macd_ema_exp_inclination, macd_ema_sin_inclination, macd_ema_pass_zero, macd_ema_pass_zero_tw, signal_ema_momentum,\
        macd_hist_ema_momentum, macd_hist_ema_abs, macd_hist_ema_tanh_inclination, macd_hist_ema_exp_inclination, macd_hist_ema_sin_inclination, macd_hist_ema_pass_zero, macd_hist_ema_pass_zero_tw,\
        stoch_hist_momentum, stoch_hist_abs, stoch_hist_tanh_inclination, stoch_hist_exp_inclination, stoch_hist_sin_inclination, stoch_hist_pass_zero, stoch_hist_pass_zero_tw,\
        trix_hist_momentum, trix_hist_abs, trix_hist_tanh_inclination, trix_hist_exp_inclination, trix_hist_sin_inclination, trix_hist_pass_zero, trix_hist_pass_zero_tw,\
        sonar_pass_zero, sonar_pass_zero_tw,\
        sonar_hist_momentum, sonar_hist_abs, sonar_hist_tanh_inclination, sonar_hist_exp_inclination, sonar_hist_sin_inclination, sonar_hist_pass_zero, sonar_hist_pass_zero_tw,\
        sigma_hist_momentum, sigma_hist_abs, sigma_hist_tanh_inclination, sigma_hist_exp_inclination, sigma_hist_sin_inclination, sigma_hist_pass_zero, sigma_hist_pass_zero_tw,\
        mfi_momentum, mfi_tanh_inclination, mfi_exp_inclination, mfi_sin_inclination,\
        mfi_hist_momentum, mfi_hist_abs, mfi_hist_tanh_inclination, mfi_hist_exp_inclination, mfi_hist_sin_inclination, mfi_hist_pass_zero, mfi_hist_pass_zero_tw,\
        lrs_momentum, lrs_abs, lrs_tanh_inclination, lrs_exp_inclination, lrs_sin_inclination,\
        lrs_hist_momentum, lrs_hist_abs, lrs_hist_tanh_inclination, lrs_hist_exp_inclination, lrs_hist_sin_inclination, lrs_hist_pass_zero, lrs_hist_pass_zero_tw,\
        lrs_rate_of_change_momentum, lrs_rate_of_change_abs, lrs_rate_of_change_tanh_inclination, lrs_rate_of_change_exp_inclination, lrs_rate_of_change_sin_inclination,\
        lrs_rate_of_change_hist_momentum, lrs_rate_of_change_hist_abs, lrs_rate_of_change_hist_tanh_inclination, lrs_rate_of_change_hist_exp_inclination, lrs_rate_of_change_hist_sin_inclination, lrs_rate_of_change_hist_pass_zero, lrs_rate_of_change_hist_pass_zero_tw,\
        r_square_momentum, r_square_tanh_inclination, r_square_exp_inclination, r_square_sin_inclination,\
        r_square_hist_momentum, r_square_hist_abs, r_square_hist_tanh_inclination, r_square_hist_exp_inclination, r_square_hist_sin_inclination, r_square_hist_pass_zero, r_square_hist_pass_zero_tw,\
        atr_change_rate_momentum, atr_change_rate_abs, atr_change_rate_tanh_inclination, atr_change_rate_exp_inclination, atr_change_rate_sin_inclination,\
        atr_change_rate_hist_momentum, atr_change_rate_hist_abs, atr_change_rate_hist_tanh_inclination, atr_change_rate_hist_exp_inclination, atr_change_rate_hist_sin_inclination, atr_change_rate_hist_pass_zero, atr_change_rate_hist_pass_zero_tw,\
        dema_se_momentum, dema_se_tanh_inclination, dema_se_exp_inclination, dema_se_sin_inclination,\
        tema_se_momentum, tema_se_tanh_inclination, tema_se_exp_inclination, tema_se_sin_inclination,\
        close_change_rate_momentum, close_change_rate_abs, close_change_rate_tanh_inclination, close_change_rate_exp_inclination, close_change_rate_sin_inclination,\
        volume_change_rate_momentum, volume_change_rate_abs, volume_change_rate_tanh_inclination, volume_change_rate_exp_inclination, volume_change_rate_sin_inclination,\
        box_and_mfi_50_pass_zero, box_and_mfi_50_pass_zero_tw, cci_X_close_change_rate\
    ]

    ################################################ Nasdaq index ########################################################################
    macd_nasdaq, signal_line_nasdaq, macd_hist_nasdaq = tech_indicator.return_macd_signal_macdhist_use_ema_cal_series_version(ser=df.nasdaq_index, short_=12, long_=26, signal_win=9)
    macd_nasdaq = macd_nasdaq / (df.nasdaq_index.values + 1e-10)
    signal_line_nasdaq = signal_line_nasdaq / (df.nasdaq_index.values + 1e-10)
    macd_hist_nasdaq = macd_hist_nasdaq / (df.nasdaq_index.values + 1e-10)

    macd_nasdaq_momentum = tech_indicator.return_momentum(ser=macd_nasdaq, before_temrs=1)
    macd_nasdaq_abs = abs(macd_nasdaq)
    macd_nasdaq_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_nasdaq, before_terms=1)
    macd_nasdaq_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_nasdaq, before_terms=1)
    macd_nasdaq_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_nasdaq, before_terms=1)

    macd_hist_nasdaq_momentum = tech_indicator.return_momentum(ser=macd_hist_nasdaq, before_temrs=1)
    macd_hist_nasdaq_abs = abs(macd_hist_nasdaq)
    macd_hist_nasdaq_tanh_inclination = tech_indicator.return_tanh_inclination(ser=macd_hist_nasdaq, before_terms=1)
    macd_hist_nasdaq_exp_inclination = tech_indicator.return_exp_inclination(ser=macd_hist_nasdaq, before_terms=1)
    macd_hist_nasdaq_sin_inclination = tech_indicator.return_sin_inclination(ser=macd_hist_nasdaq, before_terms=1)

    nasdaq_change_rate = tech_indicator.return_increase_rate(ser=df.nasdaq_index, before_terms=1)

    nasdaq_change_rate_momentum = tech_indicator.return_momentum(ser=nasdaq_change_rate, before_temrs=1)
    nasdaq_change_rate_abs = abs(nasdaq_change_rate)
    nasdaq_change_rate_tanh_inclination = tech_indicator.return_tanh_inclination(ser=nasdaq_change_rate, before_terms=1)
    nasdaq_change_rate_exp_inclination = tech_indicator.return_exp_inclination(ser=nasdaq_change_rate, before_terms=1)
    nasdaq_change_rate_sin_inclination = tech_indicator.return_sin_inclination(ser=nasdaq_change_rate, before_terms=1)

    nasdaq_feature_arrs=[
        macd_nasdaq, signal_line_nasdaq, macd_hist_nasdaq,\
        macd_nasdaq_momentum, macd_nasdaq_abs, macd_nasdaq_tanh_inclination, macd_nasdaq_exp_inclination, macd_nasdaq_sin_inclination,\
        macd_hist_nasdaq_momentum, macd_hist_nasdaq_abs, macd_hist_nasdaq_tanh_inclination, macd_hist_nasdaq_exp_inclination, macd_hist_nasdaq_sin_inclination,\
        nasdaq_change_rate,\
        nasdaq_change_rate_momentum, nasdaq_change_rate_abs, nasdaq_change_rate_tanh_inclination, nasdaq_change_rate_exp_inclination, nasdaq_change_rate_sin_inclination\
    ]

    tot_seireses = basic_feature_arrs + feature_enginnered_arrs + nasdaq_feature_arrs
    tot_col_name_list = label_col_list + tot_col_names

    if (data_split_type=='simulate'):

        train_input, train_target, val_input, val_label = support.return_merge_df(label_arr=increase_rate_arr, series_list=tot_seireses, cut_index=[tech_indicator.have_to_cut_num, profit_terms[-1]],\
            columns=tot_col_name_list, data_split_type=data_split_type)

        return train_input, train_target, val_input, val_label.reshape(-1)

    elif (data_split_type=='trade'):

        train_input, train_target, pred_input = support.return_merge_df(label_arr=increase_rate_arr, series_list=tot_seireses, cut_index=[tech_indicator.have_to_cut_num, profit_terms[-1]],\
            columns=tot_col_name_list, data_split_type=data_split_type)

        return train_input, train_target, pred_input
    
    elif (data_split_type=='model_test'):

        train_input, train_target = support.return_merge_df(label_arr=increase_rate_arr, series_list=tot_seireses, cut_index=[tech_indicator.have_to_cut_num, profit_terms[-1]],\
            columns=tot_col_name_list, data_split_type=data_split_type)

        return train_input, train_target

    else:
        print("Worng data_split_type")
        return -1
