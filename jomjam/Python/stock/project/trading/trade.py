from utils import loading_data, tech_indicator, preprocessing, quantile_df, feature_selection
from trading import model_part

import pandas as pd
import numpy as np


def return_buy_codebook_and_prediction(code_list, last_date, term_for_cal, day_for_buy, model_opts, feature_cate_quantile_num=0, feature_selection_opt=False):
    #For output
    order_list=[]
    pred_list=[]

    for code in code_list:
        #Get stock data
        reload_df = loading_data.return_default_df(code=code, trading_date_int=last_date, add_length=term_for_cal+day_for_buy[-1])
        #Checking length
        if ( reload_df.shape[0] < (term_for_cal + day_for_buy[-1] + tech_indicator.have_to_cut_num) ):
            print("Shortage data / Code : {} / Shape : {}".format(code, reload_df.shape))
            continue
        
        #Preprocessing df
        train_input, train_target, pred_input = preprocessing.return_preprocessed_df(df=reload_df, profit_terms=day_for_buy, data_split_type='trade')
        train_input = pd.DataFrame(train_input, columns=preprocessing.tot_col_names)
        val_input = pd.DataFrame(val_input, columns=preprocessing.tot_col_names)

        #Change to categorical (bins) df
        if (feature_cate_quantile_num!=0):
            train_input, val_input = quantile_df.return_quantile_df(df=train_input, quantile_num=feature_cate_quantile_num, valid_opt=True, valid_set=pred_input)
            train_input = train_input.astype('category')
            val_input = val_input.astype('category')

        #For cum day result
        pred_dd_list=[]
        rmse_dd_list=[]

        #Loop for day list
        for idx in range(len(day_for_buy)):
            target_for_feature_selection = train_target[:,idx]

            if feature_selection_opt :

                filtered_columns_split_inputs = feature_selection.return_filtered_feature_df(train_input=train_input, train_target=target_for_feature_selection,\
                        feature_cate_quantile_num=feature_cate_quantile_num)
                    
                selected_col_list = list(filtered_columns_split_inputs.columns)
                input_for_train_in_idx = filtered_columns_split_inputs
                input_for_valid_in_idx = val_input[selected_col_list]

            else:

                selected_col_list = list(preprocessing.tot_col_names)
                input_for_train_in_idx = train_input
                input_for_valid_in_idx = val_input

            #Loop for model list
            fin_rate_tot = 0
            fin_rate_rmse = 0
            fin_rate_num = 0
            for mod_op in model_opts:

                fin_rate, rmse_ = model_part.get_result_from_model(train_input=input_for_train_in_idx, train_target=target_for_feature_selection,\
                        running_opt='trade', val_input=input_for_valid_in_idx, model_opt=mod_op,\
                        feature_cate_quantile_num=feature_cate_quantile_num, selected_col=selected_col_list)

                fin_rate_tot += fin_rate
                fin_rate_rmse += rmse_
                fin_rate_num += 1
            
            pred_dd_list.append( fin_rate_tot / fin_rate_num )
            rmse_dd_list.append( fin_rate_rmse / fin_rate_num )

        
        pred_dd_arr = np.array(pred_dd_list)
        rmse_dd_arr = np.array(rmse_dd_list)

        #Filter code
        if (pred_dd_arr.mean()>=1.01):
            order_list.append(code)
            pred_list.append(pred_dd_arr-rmse_dd_arr)

            print("Code : {} / Prediction : {} / RMSE : {}".format(code, pred_dd_arr, rmse_dd_arr))

        else:
            print("Code {} is Not enough rate....".format(code))

    ###### Sorting...
    order_arr = np.array(order_list)
    pred_arr = np.array(pred_list)

    pred_index = pred_arr.mean(axis=1).argsort()

    order_arr = order_arr[pred_index][-5:]
    pred_arr = pred_arr[pred_index][-5:]

    print("#################################################################################")
    print("Filtered stock is : {}".format(order_arr))
    print("=========== Predicted result (increase rate) is (Matrix) ===========")
    print(pred_arr)
    print("====================================================================")
    return order_arr, pred_arr
        