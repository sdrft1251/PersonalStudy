
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from utils import loading_data, preprocessing, feature_selection, quantile_df
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor




def return_val_pred(train_input, train_target, selected_col, train_size=0.7, feature_cate_quantile_num=0):
    train_cut_index = int(len(train_target)*train_size)

    train_parts_input = train_input.iloc[:train_cut_index]
    train_parts_target = train_target[:train_cut_index]
    val_parts_input = train_input.iloc[train_cut_index:]

    if (feature_cate_quantile_num==0):
        scaler = RobustScaler()
        scaler.fit(train_parts_input)
        train_parts_input = scaler.transform(train_parts_input)
        val_parts_input = scaler.transform(val_parts_input)
    
    model_ob = LGBMRegressor(objective='regression', num_iterations=10**3)
    if (feature_cate_quantile_num!=0):
        model_ob.fit(train_parts_input, train_parts_target, eval_set=[(train_parts_input, train_parts_target)], early_stopping_rounds=100, verbose=False, categorical_feature=selected_col)
    else:
        model_ob.fit(train_parts_input, train_parts_target, eval_set=[(train_parts_input, train_parts_target)], early_stopping_rounds=100, verbose=False)

    val_pred = model_ob.predict(val_parts_input)
    return val_pred



def return_model_result(code_list, testing_date, terms, profit_terms, train_size, feature_selection_opt, feature_cate_quantile_num=0):
    tot_num=0
    code_start=True
    for code in code_list:
        reload_df = loading_data.return_default_df(code=code, trading_date_int=testing_date, add_length=terms)
        if ( reload_df.shape[0] < (terms + tech_indicator.have_to_cut_num) ):
            print("Shortage data / Code : {} / Shape : {}".format(code, reload_df.shape))
            continue
        #Preprocessing Data Set
        train_input, train_target = preprocessing.return_preprocessed_df(df=reload_df, profit_terms=profit_terms, data_split_type='model_test')
        train_input = pd.DataFrame(train_input, columns=preprocessing.tot_col_names)
        
        if (feature_cate_quantile_num!=0):
            train_input = quantile_df.return_quantile_df(df=train_input, quantile_num=feature_cate_quantile_num)
            train_input = train_input.astype('category')

        print("Code : {}".format(code))
        
        val_cum=[]
        start=True
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

            val_pred_get = return_val_pred(train_input=inputs_for_get_rmse, train_target=target_for_feature_selection, selected_col=selected_col_list,\
                train_size=train_size, feature_cate_quantile_num=feature_cate_quantile_num)
            val_pred_get = np.array(val_pred_get).reshape(-1,1)
            if start:
                val_cum = val_pred_get
                start=False
            else:
                val_cum = np.concatenate((val_cum, val_pred_get), axis=1)
        
        val_cum_arr = np.array(val_cum)
        val_cum_arr_filter_max = val_cum_arr.max(axis=1)
        val_cum_arr_filter_max = val_cum_arr_filter_max.reshape(-1,1)
        val_cum_arr_filter_mean = val_cum_arr.mean(axis=1)
        val_cum_arr_filter_mean = val_cum_arr_filter_mean.reshape(-1,1)
        val_cum_arr_filter_std = val_cum_arr.std(axis=1)
        val_cum_arr_filter_std = val_cum_arr_filter_std.reshape(-1,1)

        train_cut_index = int(len(train_target)*train_size)
        val_parts_target = train_target[train_cut_index:]
        val_parts_target_max = val_parts_target.max(axis=1).reshape(-1,1)
        val_parts_target_mean = val_parts_target.mean(axis=1).reshape(-1,1)

        val_part_tot = np.concatenate((val_cum_arr_filter_max, val_cum_arr_filter_mean, val_cum_arr_filter_std, val_cum_arr,\
        val_parts_target_max, val_parts_target_mean, val_parts_target), axis=1)

        col_names=['pred_max', 'pred_mean', 'pred_std']
        for temr_ in profit_terms:
            col_names.append("pred_after_day_"+str(temr_))
        col_names.append('Real_target_max')
        col_names.append('Real_target_mean')
        for temr_ in profit_terms:
            col_names.append("increase_rate_after_day_"+str(temr_))

        fin_result = pd.DataFrame(val_part_tot, columns=col_names)
        corr_df = fin_result.corr()
        corr_df_select = corr_df.iloc[-len(profit_terms)-2:,:-len(profit_terms)-2]

        print(corr_df_select)

        if code_start:
            sum_corr_coef_df = corr_df_select
            tot_num+=1
            code_start=False
        else:
            val_sum = sum_corr_coef_df.values + corr_df_select.values
            tot_num+=1
            sum_corr_coef_df = pd.DataFrame(val_sum, columns=sum_corr_coef_df.columns.values, index=sum_corr_coef_df.index.values)

    fin_arr = sum_corr_coef_df.values / tot_num
    mean_corr_coef_df = pd.DataFrame(fin_arr, columns=sum_corr_coef_df.columns.values, index=sum_corr_coef_df.index.values)

    print("Final....")
    print(mean_corr_coef_df)
    return mean_corr_coef_df
     




def return_model_result_with_terms_list(code_list, testing_date, terms_list, profit_terms, train_size, feature_selection_opt, feature_cate_quantile_num=0):
    tot_num=0
    code_start=True
    for code in code_list:
        for terms in terms_list:
            reload_df = loading_data.return_default_df(code=code, trading_date_int=testing_date, add_length=terms)
            if ( reload_df.shape[0] < (terms + tech_indicator.have_to_cut_num) ):
                print("Shortage data / Code : {} / Shape : {}".format(code, reload_df.shape))
                continue
            #Preprocessing Data Set
            train_input, train_target = preprocessing.return_preprocessed_df(df=reload_df, profit_terms=profit_terms, data_split_type='model_test')
            train_input = pd.DataFrame(train_input, columns=preprocessing.tot_col_names)
            
            if (feature_cate_quantile_num!=0):
                train_input = quantile_df.return_quantile_df(df=train_input, quantile_num=feature_cate_quantile_num)
                train_input = train_input.astype('category')

            print("Code : {} / terms : {}".format(code, terms))
            
            val_cum=[]
            start=True
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

                val_pred_get = return_val_pred(train_input=inputs_for_get_rmse, train_target=target_for_feature_selection, selected_col=selected_col_list,\
                    train_size=train_size, feature_cate_quantile_num=feature_cate_quantile_num)
                val_pred_get = np.array(val_pred_get).reshape(-1,1)
                if start:
                    val_cum = val_pred_get
                    start=False
                else:
                    val_cum = np.concatenate((val_cum, val_pred_get), axis=1)
            
            val_cum_arr = np.array(val_cum)
            val_cum_arr_filter_max = val_cum_arr.max(axis=1)
            val_cum_arr_filter_max = val_cum_arr_filter_max.reshape(-1,1)
            val_cum_arr_filter_mean = val_cum_arr.mean(axis=1)
            val_cum_arr_filter_mean = val_cum_arr_filter_mean.reshape(-1,1)
            val_cum_arr_filter_std = val_cum_arr.std(axis=1)
            val_cum_arr_filter_std = val_cum_arr_filter_std.reshape(-1,1)

            train_cut_index = int(len(train_target)*train_size)
            val_parts_target = train_target[train_cut_index:]
            val_parts_target_max = val_parts_target.max(axis=1).reshape(-1,1)
            val_parts_target_mean = val_parts_target.mean(axis=1).reshape(-1,1)

            val_part_tot = np.concatenate((val_cum_arr_filter_max, val_cum_arr_filter_mean, val_cum_arr_filter_std, val_cum_arr,\
            val_parts_target_max, val_parts_target_mean, val_parts_target), axis=1)

            col_names=['pred_max', 'pred_mean', 'pred_std']
            for temr_ in profit_terms:
                col_names.append("pred_after_day_"+str(temr_))
            col_names.append('Real_target_max')
            col_names.append('Real_target_mean')
            for temr_ in profit_terms:
                col_names.append("increase_rate_after_day_"+str(temr_))

            fin_result = pd.DataFrame(val_part_tot, columns=col_names)
            corr_df = fin_result.corr()
            corr_df_select = corr_df.iloc[-len(profit_terms)-2:,:-len(profit_terms)-2]

            print(corr_df_select)

            if code_start:
                sum_corr_coef_df = corr_df_select
                tot_num+=1
                code_start=False
            else:
                val_sum = sum_corr_coef_df.values + corr_df_select.values
                tot_num+=1
                sum_corr_coef_df = pd.DataFrame(val_sum, columns=sum_corr_coef_df.columns.values, index=sum_corr_coef_df.index.values)

    fin_arr = sum_corr_coef_df.values / tot_num
    mean_corr_coef_df = pd.DataFrame(fin_arr, columns=sum_corr_coef_df.columns.values, index=sum_corr_coef_df.index.values)

    print("Final....")
    print(mean_corr_coef_df)
    return mean_corr_coef_df