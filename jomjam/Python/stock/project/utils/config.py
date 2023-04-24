##### DB file path #####
db_path = "./data/kospi_past_day_data_set.db"
########################

##### Total Columns #####
feats = [\
        'macd_ma', 'signal_line_ma', 'macd_hist_ma', 'macd_ema', 'signal_line_ema', 'macd_hist_ema',\
        'rsi', 'stoch_k', 'stoch_d', 'stoch_hist', 'bol_rate', 'volume_trend_incline', 'trix', 'trix_signal', 'trix_hist',\
        'sonar', 'signar', 'sonar_hist', 'sigma', 'sigma_signal', 'sigma_hist', 'mfi', 'mfi_signal',\
        'lrs', 'lrs_signal', 'lrs_hist', 'lrs_rate_of_change', 'lrs_rate_of_change_signal', 'lrs_rate_of_change_hist', 'r_square', 'r_square_signal',\
        'atr_change_rate', 'atr_change_rate_signal', 'atr_change_rate_hist',\
        'dema_se', 'tema_se', 'close_change_rate', 'volume_change_rate',\
        'macd_ma_momentum', 'signal_ma_momentum', 'macd_hist_ma_momentum', 'macd_ma_abs', 'macd_ma_tanh_inclination', 'macd_ma_exp_inclination', 'macd_hist_ma_exp_inclination', 'macd_hist_ma_tanh_inclination',\
        'macd_ema_momentum', 'signal_ema_momentum', 'macd_hist_ema_momentum', 'macd_ema_abs', 'macd_ema_tanh_inclination', 'macd_ema_exp_inclination', 'macd_hist_ema_tanh_inclination', 'macd_hist_ema_exp_inclination',\
        'macd_nasdaq', 'signal_line_nasdaq', 'macd_hist_nasdaq', 'macd_nasdaq_momentum', 'macd_hist_nasdaq_momentum', 'macd_nasdaq_abs', 'macd_nasdaq_tanh_inclination', 'macd_nasdaq_exp_inclination',\
        'macd_hist_nasdaq_tanh_inclination', 'macd_hist_nasdaq_exp_inclination', 'nasdaq_change_rate',\
        'macd_hist_ma_abs', 'macd_hist_ema_abs',\
        'stoch_hist_momentum', 'stoch_hist_abs', 'stoch_hist_tanh_inclination', 'stoch_hist_exp_inclination',\
        'trix_hist_momentum', 'trix_hist_abs', 'trix_hist_tanh_inclination', 'trix_hist_exp_inclination',\
        'sonar_hist_momentum', 'sonar_hist_abs', 'sonar_hist_tanh_inclination', 'sonar_hist_exp_inclination',\
        'sigma_hist_momentum', 'sigma_hist_abs', 'sigma_hist_tanh_inclination', 'sigma_hist_exp_inclination',\
        'mfi_momentum', 'mfi_tanh_inclination', 'mfi_exp_inclination',\
        'mfi_hist_momentum', 'mfi_hist_abs', 'mfi_hist_tanh_inclination', 'mfi_hist_exp_inclination',\
        'lrs_momentum', 'lrs_abs', 'lrs_tanh_inclination', 'lrs_exp_inclination',\
        'lrs_hist_momentum', 'lrs_hist_abs', 'lrs_hist_tanh_inclination', 'lrs_hist_exp_inclination',\
        'lrs_rate_of_change_momentum', 'lrs_rate_of_change_abs', 'lrs_rate_of_change_tanh_inclination', 'lrs_rate_of_change_exp_inclination',\
        'lrs_rate_of_change_hist_momentum', 'lrs_rate_of_change_hist_abs', 'lrs_rate_of_change_hist_tanh_inclination', 'lrs_rate_of_change_hist_exp_inclination',\
        'r_square_momentum', 'r_square_tanh_inclination', 'r_square_exp_inclination',\
        'r_square_hist_momentum', 'r_square_hist_abs', 'r_square_hist_tanh_inclination', 'r_square_hist_exp_inclination',\
        'atr_change_rate_momentum', 'atr_change_rate_abs', 'atr_change_rate_tanh_inclination', 'atr_change_rate_exp_inclination',\
        'atr_change_rate_hist_momentum', 'atr_change_rate_hist_abs', 'atr_change_rate_hist_tanh_inclination', 'atr_change_rate_hist_exp_inclination',\
        'dema_se_momentum', 'dema_se_tanh_inclination', 'dema_se_exp_inclination',\
        'tema_se_momentum', 'tema_se_tanh_inclination', 'tema_se_exp_inclination',\
        'close_change_rate_momentum', 'close_change_rate_abs', 'close_change_rate_tanh_inclination', 'close_change_rate_exp_inclination',\
        'volume_change_rate_momentum', 'volume_change_rate_abs', 'volume_change_rate_tanh_inclination', 'volume_change_rate_exp_inclination',\
        'macd_hist_nasdaq_abs', 'nasdaq_change_rate_momentum', 'nasdaq_change_rate_abs', 'nasdaq_change_rate_tanh_inclination', 'nasdaq_change_rate_exp_inclination'\
        ]
#########################



##### For Preprocess ####
preprocessed_df_front_cut_index=33
#########################