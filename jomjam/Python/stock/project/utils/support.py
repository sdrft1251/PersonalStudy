import pandas as pd
import numpy as np



def return_merge_df(label_arr, series_list, cut_index, columns, data_split_type):
    start=True
    for ser in series_list:
        arr = ser.reshape(-1,1)
        if start:
            result_arr = np.concatenate((label_arr, arr), axis=1)
            start=False
        else:
            result_arr = np.concatenate((result_arr, arr), axis=1)
    
    result_df = pd.DataFrame(result_arr, columns=columns)
    if (data_split_type=='simulate'):
        train_input = result_df.values[cut_index[0]:-(cut_index[1]*2),label_arr.shape[1]:]
        train_target = result_df.values[cut_index[0]:-(cut_index[1]*2),:label_arr.shape[1]]

        val_input = result_df.values[-(2+cut_index[1]):-cut_index[1],label_arr.shape[1]:]
        val_label = result_df.values[-(1+cut_index[1]):-cut_index[1],:label_arr.shape[1]]

        return train_input, train_target, val_input, val_label
    elif (data_split_type=='trade'):
        train_input = result_df.values[cut_index[0]:-cut_index[1] , label_arr.shape[1]:]
        train_target = result_df.values[cut_index[0]:-cut_index[1] , :label_arr.shape[1]]

        pred_input = result_df.values[-2:,label_arr.shape[1]:]
        return train_input, train_target, pred_input

    elif (data_split_type=='model_test'):
        train_input = result_df.values[cut_index[0]:-cut_index[1] , label_arr.shape[1]:]
        train_target = result_df.values[cut_index[0]:-cut_index[1] , :label_arr.shape[1]]
        return train_input, train_target
    else:
        print("Worng data_split_type")
        return -1




