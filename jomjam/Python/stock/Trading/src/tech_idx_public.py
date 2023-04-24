import numpy as np

def dmi(high_arr, low_arr, close_arr, window=1):
    after_high_arr = np.roll(high_arr, -1)[:-1]
    after_low_arr = np.roll(low_arr, -1)[:-1]
    after_close_arr = np.roll(close_arr, -1)[:-1]

    high_arr = high_arr[:-1]
    low_arr = low_arr[:-1]
    close_arr = close_arr[:-1]
    # TR 연산
    ar_1 = abs(after_high_arr - after_low_arr).reshape(-1,1)
    ar_2 = abs(after_high_arr - close_arr).reshape(-1,1)
    ar_3 = abs(after_low_arr - close_arr).reshape(-1,1)
    tr_arr = np.concatenate((ar_1, ar_2, ar_3), axis=1).max(axis=1)
    # DM 연산
    ar_1 = (after_high_arr - high_arr)
    ar_2 = (low_arr - after_low_arr)
    mask_both_minus = np.where(np.logical_and(ar_1<0, ar_2<0), 1, 0)
    dm_arr = np.where(ar_1>=ar_2, ar_1, -ar_2)
    dm_arr = np.where(mask_both_minus==1, 0, dm_arr)
    
    if window>=2:
        # 이동평균 구하기
        tr_ma = moving_average_for_numpy(arr=tr_arr, window=window)
        dm_ma = moving_average_for_numpy(arr=dm_arr, window=window)
        tr_ma = np.where(tr_ma==0, 1, tr_ma)
        return dm_ma / tr_ma
    else:
        tr_arr = np.where(tr_arr==0, 1, tr_arr)
        return dm_arr / tr_arr