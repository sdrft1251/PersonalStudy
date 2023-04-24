import pandas as pd
import numpy as np




def return_quantile_df(df, quantile_num, valid_opt=False, valid_set=0):
    if (100%quantile_num!=0):
        print("Wrong quantile_num")
        return -1
    quan_range = 100//quantile_num
    quan_range_list=[]
    tmp=0
    while(tmp<=100):
        quan_range_list.append(tmp)
        tmp += quan_range
    start=True
    for col in df.columns:
        arr = df[col].values
        percentile_arr = np.percentile(arr, quan_range_list, interpolation='nearest')

        new_arr = np.zeros(len(arr))

        ############################## For valid set
        if valid_opt:
            arr_val = valid_set[col].values
            new_arr_val = np.zeros(len(arr_val))
        ##############################

        for idx in range(len(percentile_arr)-1):
            new_arr = np.where((arr>=percentile_arr[idx])&(arr<=percentile_arr[idx+1]), idx, new_arr)

            ############################## For valid set
            if valid_opt:
                new_arr_val = np.where((arr_val>=percentile_arr[idx])&(arr_val<=percentile_arr[idx+1]), idx, new_arr_val)
            ##############################
        
        if start:
            for_concat_arr = new_arr.reshape(-1,1)
            

            ############################## For valid set
            if valid_opt:
                for_concat_arr_val = new_arr_val.reshape(-1,1)
            ##############################
            
            start=False

        else:
            for_concat_arr = np.concatenate((for_concat_arr, new_arr.reshape(-1,1)), axis=1)

            ############################## For valid set
            if valid_opt:
                for_concat_arr_val = np.concatenate((for_concat_arr_val, new_arr_val.reshape(-1,1)), axis=1)
            ##############################
    
    new_df = pd.DataFrame(for_concat_arr, columns=df.columns.values)

    ############################## For valid set
    if valid_opt:
        new_df_val = pd.DataFrame(for_concat_arr_val, columns=df.columns.values)
        return new_df, new_df_val
    ##############################

    return new_df
