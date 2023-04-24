import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import deque

def return_data_set(rev_df, inv_df, cut_date=20120101):
    rev_df.drop(columns=['sign'], inplace=True)
    inv_df.drop(columns=['sign'], inplace=True)

    rev_df = rev_df[rev_df.date>=cut_date]
    inv_df = inv_df[inv_df.date>=cut_date]

    #===== Mean price =====
    rev_df["5_mean"] = rev_df.close.rolling(window=5).mean()
    inv_df["5_mean"] = inv_df.close.rolling(window=5).mean()
    rev_df["20_mean"] = rev_df.close.rolling(window=20).mean()
    inv_df["20_mean"] = inv_df.close.rolling(window=20).mean()
    rev_df["60_mean"] = rev_df.close.rolling(window=60).mean()
    inv_df["60_mean"] = inv_df.close.rolling(window=60).mean()

    rev_df.dropna(inplace=True)
    inv_df.dropna(inplace=True)

    #===== Difference with mean price =====
    rev_df["diff_with_5_mean"] = rev_df["close"].values - rev_df["5_mean"].values
    rev_df["diff_with_20_mean"] = rev_df["close"].values - rev_df["20_mean"].values
    rev_df["diff_with_60_mean"] = rev_df["close"].values - rev_df["60_mean"].values

    inv_df["diff_with_5_mean"] = inv_df["close"].values - inv_df["5_mean"].values
    inv_df["diff_with_20_mean"] = inv_df["close"].values - inv_df["20_mean"].values
    inv_df["diff_with_60_mean"] = inv_df["close"].values - inv_df["60_mean"].values

    #===== Calculate RSI =====
    tmp_val_arr = rev_df.increase.values
    tmp_val_arr = np.where(tmp_val_arr>=0, tmp_val_arr, 0)
    rev_df["increase_U_price"] = tmp_val_arr
    tmp_val_arr = rev_df.increase.values
    tmp_val_arr = np.where(tmp_val_arr<0, -tmp_val_arr, 0)
    rev_df["increase_D_price"] = tmp_val_arr

    tmp_val_arr = inv_df.increase.values
    tmp_val_arr = np.where(tmp_val_arr>=0, tmp_val_arr, 0)
    inv_df["increase_U_price"] = tmp_val_arr
    tmp_val_arr = inv_df.increase.values
    tmp_val_arr = np.where(tmp_val_arr<0, -tmp_val_arr, 0)
    inv_df["increase_D_price"] = tmp_val_arr

    rev_df["AU"] = rev_df.increase_U_price.rolling(window=14).mean()
    rev_df["AD"] = rev_df.increase_D_price.rolling(window=14).mean()
    inv_df["AU"] = inv_df.increase_U_price.rolling(window=14).mean()
    inv_df["AD"] = inv_df.increase_D_price.rolling(window=14).mean()

    rev_df.dropna(inplace=True)
    inv_df.dropna(inplace=True)

    rev_df["RSI"] = rev_df["AU"].values / (rev_df["AU"].values + rev_df["AD"].values)
    inv_df["RSI"] = inv_df["AU"].values / (inv_df["AU"].values + inv_df["AD"].values)

    #===== Candle information =====
    rev_df["high_low_range"] = rev_df["high"].values - rev_df["low"].values
    inv_df["high_low_range"] = inv_df["high"].values - inv_df["low"].values

    rev_df["open_rate"] = (rev_df["open"].values - rev_df["low"].values) / (rev_df["high_low_range"].values + 1e-10)
    rev_df["close_rate"] = (rev_df["close"].values - rev_df["low"].values) / (rev_df["high_low_range"].values + 1e-10)
    inv_df["open_rate"] = (inv_df["open"].values - inv_df["low"].values) / (inv_df["high_low_range"].values + 1e-10)
    inv_df["close_rate"] = (inv_df["close"].values - inv_df["low"].values) / (inv_df["high_low_range"].values + 1e-10)

    #===== Select Columns =====
    #"open","high","low","close","marketcap","numstock","corpbuycum","5_mean","20_mean","60_mean","increase_U_price","increase_D_price"
    #result_rev_df = rev_df[["date","volume","amount","foreign","corpbuy","diff_with_5_mean","diff_with_20_mean","diff_with_60_mean","increase","AU","AD","RSI","high_low_range","open_rate","close_rate"]]
    #result_inv_df = inv_df[["date","volume","amount","foreign","corpbuy","diff_with_5_mean","diff_with_20_mean","diff_with_60_mean","increase","AU","AD","RSI","high_low_range","open_rate","close_rate"]]
    rev_df_for_diff = rev_df[["date","open","high","low","close","volume","amount","foreign","corpbuy"]]
    inv_df_for_diff = inv_df[["date","open","high","low","close","volume","amount","foreign","corpbuy"]]
    rev_df_not_diff = rev_df[["date","diff_with_5_mean","diff_with_20_mean","diff_with_60_mean","increase","increase_U_price","increase_D_price","AU","AD","RSI","high_low_range","open_rate","close_rate"]]
    inv_df_not_diff = inv_df[["date","diff_with_5_mean","diff_with_20_mean","diff_with_60_mean","increase","increase_U_price","increase_D_price","AU","AD","RSI","high_low_range","open_rate","close_rate"]]

    #result_rev_df = rev_df.copy()
    #result_inv_df = inv_df.copy()

    #===== Target =====
    for_target_rev_df = pd.DataFrame()
    for_target_inv_df = pd.DataFrame()
    for_target_rev_df["close"] = rev_df.close.values
    for_target_inv_df["close"] = inv_df.close.values

    for_target_rev_df["day_after_close"] = for_target_rev_df.close.shift(periods=-1)
    for_target_inv_df["day_after_close"] = for_target_inv_df.close.shift(periods=-1)

    for_target_rev_df.dropna(inplace=True)
    for_target_inv_df.dropna(inplace=True)

    for_target_rev_df["close_rate"] = for_target_rev_df["day_after_close"].values / for_target_rev_df["close"].values
    for_target_inv_df["close_rate"] = for_target_inv_df["day_after_close"].values / for_target_inv_df["close"].values

    tmp_val_arr_rev = for_target_rev_df.close_rate.values
    tmp_val_arr_inv = for_target_inv_df.close_rate.values

    target_arr = np.zeros(len(tmp_val_arr_rev))
    target_arr = np.where(tmp_val_arr_rev>=1.01, 1, target_arr)
    target_arr = np.where(tmp_val_arr_rev<=0.995, 2, target_arr)
    #target_arr = np.where(tmp_val_arr_inv>1.01, 2, target_arr)

    #===== Input =====
    rev_df_for_diff.reset_index(inplace=True, drop=True)
    inv_df_for_diff.reset_index(inplace=True, drop=True)
    rev_df_not_diff.reset_index(inplace=True, drop=True)
    inv_df_not_diff.reset_index(inplace=True, drop=True)

    rev_df_for_diff = rev_df_for_diff.iloc[:-1]
    inv_df_for_diff = inv_df_for_diff.iloc[:-1]
    rev_df_not_diff = rev_df_not_diff.iloc[:-1]
    inv_df_not_diff = inv_df_not_diff.iloc[:-1]

    merge_df_for_diff = pd.merge(rev_df_for_diff, inv_df_for_diff, how='left', left_on='date', right_on='date', suffixes=('_rev', '_inv'))
    merge_df_not_diff = pd.merge(rev_df_not_diff, inv_df_not_diff, how='left', left_on='date', right_on='date', suffixes=('_rev', '_inv'))

    merge_df_for_diff.drop(columns=["date"], inplace=True)
    merge_df_not_diff.drop(columns=["date"], inplace=True)

    merge_df_for_diff = merge_df_for_diff - merge_df_for_diff.shift(periods=1)
    merge_df_for_diff.dropna(inplace=True)
    target_arr = target_arr[1:]
    merge_df_not_diff = merge_df_not_diff.iloc[1:]

    merge_df_for_diff.reset_index(inplace=True, drop=True)
    merge_df_not_diff.reset_index(inplace=True, drop=True)

    merge_df = pd.merge(merge_df_for_diff, merge_df_not_diff, how='left', left_index=True, right_index=True, suffixes=('_fordiff', '_notdiff'))

    #merge_df = merge_df - merge_df.shift(periods=1)
    #merge_df.dropna(inplace=True)
    #target_arr = target_arr[1:]

    #merge_df = merge_df[["high_rev","increase_U_price_inv","open_rev","high_low_range_rev","increase_D_price_rev","increase_D_price_inv","open_inv","low_inv","increase_U_price_rev","close_rate_rev"]]
    print("Final Columns is =====")
    print(merge_df.columns)
    #===== Split train & valid data set =====
    train_inputs = merge_df.iloc[:int(len(merge_df)*0.9)].values
    valid_inputs = merge_df.iloc[int(len(merge_df)*0.9):].values
    train_target = target_arr[:int(len(merge_df)*0.9)]
    valid_target = target_arr[int(len(merge_df)*0.9):]

    train_inputs = np.array(train_inputs, dtype=np.float32)
    valid_inputs = np.array(valid_inputs, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    valid_target = np.array(valid_target, dtype=np.float32)

    train_inputs_mean = train_inputs.mean(axis=0)
    train_inputs_std = train_inputs.std(axis=0)

    train_inputs = (train_inputs - train_inputs_mean) / (train_inputs_std + 1e-10)
    valid_inputs = (valid_inputs - train_inputs_mean) / (train_inputs_std + 1e-10)

    #train_inputs, valid_inputs, train_target, valid_target =  train_test_split(merge_df, target_arr, test_size=0.2, stratify=target_arr)

    #===== Return data set =====
    print("~~~~~~~~Data Set Size is = Train input : {} Train target : {} Valid input : {} Valid target : {}".format(train_inputs.shape, train_target.shape, valid_inputs.shape, valid_target.shape))
    print(np.unique(train_target), np.unique(valid_target))
    return train_inputs, train_target, valid_inputs, valid_target

def return_timeseires_set(x_inputs, y_inputs, term):
    dq = deque(maxlen=term)
    x = []
    y = []
    for idx in range(len(x_inputs)-1):
        dq.append(x_inputs[idx])
        if len(dq)==term:
            inputs_ = np.array(dq, dtype=np.float32)
            x.append(inputs_)
            y.append(y_inputs[idx])


    inputs_arr = np.array(x, dtype=np.float32)
    label_arr = np.array(y, dtype=np.uint8)

    return inputs_arr, label_arr

