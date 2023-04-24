import numpy as np
from .utils import return_coef_from_closearr

###################### Moving window 용 ######################
def reg_coef_trend(arr, window):
    coef_results = []
    tot_length = len(arr)
    for idx in range(tot_length-window):
        sample = arr[idx:idx+window]
        coef = return_coef_from_closearr(sample)
        coef_results.append(coef)
    return np.array(coef_results)

###################### 개별 Array에 대한 ######################
def check_trend(arr, window):
    sample_arr = arr[-window:]
    coef = return_coef_from_closearr(sample_arr)
    return coef

