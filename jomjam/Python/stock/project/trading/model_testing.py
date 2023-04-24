import sqlite3
import numpy as np
from trading import model_part
from utils import config
import pandas as pd
from utils import preprocessing
from utils import loading_data, feature_selection, quantile_df


def return_model_result(code_list, testing_date, terms, profit_terms, train_size, model_opt, feature_selection_opt, feature_cate_quantile_num=0):
    conn = sqlite3.connect(config.db_path)
    cur = conn.cursor()
    train_rmse_arr = np.zeros(len(profit_terms))
    val_rmse_arr = np.zeros(len(profit_terms))
    tot_num=0
    for code in code_list:
        reload_df = loading_data.return_default_df(code=code, trading_date_int=testing_date, add_length=terms)
        if ( reload_df.shape[0] < (terms + 35) ):
            print("Shortage data / Code : {} / Shape : {}".format(code, reload_df.shape))
            continue
        #Preprocessing Data Set
        train_input, train_target = preprocessing.return_preprocessed_df(df=reload_df, profit_terms=profit_terms, data_split_type='model_test')
        train_input = pd.DataFrame(train_input, columns=preprocessing.tot_col_names)
        
        if (feature_cate_quantile_num!=0):
            train_input = quantile_df.return_quantile_df(df=train_input, quantile_num=feature_cate_quantile_num)
            train_input = train_input.astype('category')

        print("Code : {}".format(code))
        trian_rmse_for_print =np.zeros(len(profit_terms))
        val_rmse_for_print =np.zeros(len(profit_terms))
        for idx in range(len(profit_terms)):
            target_for_feature_selection = train_target[:,idx]
            if feature_selection_opt :

                filtered_columns_split_inputs = feature_selection.return_filtered_feature_df(train_input=train_input, train_target=target_for_feature_selection,\
                    feature_cate_quantile_num=feature_cate_quantile_num)

                inputs_for_get_rmse = filtered_columns_split_inputs
                selected_col_list = list(filtered_columns_split_inputs.columns)

            else:
                selected_col_list = list(preprocessing.tot_col_names)
                inputs_for_get_rmse = train_input
                print(len(inputs_for_get_rmse.columns), len(selected_col_list))

            train_rmse, val_rmse = model_part.get_result_from_model(train_input=inputs_for_get_rmse, train_target=target_for_feature_selection, running_opt='score', \
                train_size=train_size, model_opt=model_opt, feature_cate_quantile_num=feature_cate_quantile_num, selected_col=selected_col_list)
            print(train_rmse, val_rmse)
            train_rmse_arr[idx] += train_rmse
            val_rmse_arr[idx] += val_rmse
            trian_rmse_for_print[idx] = train_rmse
            val_rmse_for_print[idx] = val_rmse
        tot_num+=1
        print("Train rmse : {} / Val rmse : {}".format(trian_rmse_for_print, val_rmse_for_print))
    conn.close()

    train_rmse_arr = train_rmse_arr/tot_num
    val_rmse_arr = val_rmse_arr/tot_num

    print("Final!!!! Mean tot train rmse : {} / Mean tot val rmse : {}".format(train_rmse_arr, val_rmse_arr))