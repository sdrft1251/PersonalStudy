from datetime import date
from datetime import timedelta
import sqlite3
from utils import config
import pandas as pd
from utils import preprocessing, tech_indicator
from trading import model_part
import numpy as np
from utils import loading_data, feature_selection, quantile_df

def return_date_correct_format(datetime):
    datetime_str = str(datetime.year) + change_month_days(str(datetime.month)) + change_month_days(str(datetime.day))
    return int(datetime_str)
def change_month_days(time):
    if (len(time)<=1):
        time= '0'+time
    return time



def simulate(code_list, start_date, end_date, term_for_cal, day_for_buy, feature_selection_opt, model_opts, feature_cate_quantile_num=0):
    start_date_str = str(start_date)
    end_date_str = str(end_date)
    start_date_datetime = date(int(start_date_str[:4]), int(start_date_str[4:6]), int(start_date_str[6:]))
    end_date_datetime = date(int(end_date_str[:4]), int(end_date_str[4:6]), int(end_date_str[6:]))

    #Initialize book
    order_book={}
    pred_book={}
    real_book={}

    ###### For Showing
    origin_money=np.full(5, 2000)
    cum_day=0

    ###### Back Testing Start
    trading_day = start_date_datetime
    while(trading_day <= end_date_datetime):
        order_list=[]
        pred_list=[]
        real_list=[]

        #For checking weekday
        if (trading_day.weekday() in [5,6]):
            trading_day += timedelta(days=1)
            continue

        #Make Date att
        trading_date_int = return_date_correct_format(datetime=trading_day)
        #Make Date right format
        for code in code_list:

            #Loading Data
            reload_df = loading_data.return_default_df(code=code, trading_date_int=trading_date_int, add_length=term_for_cal+day_for_buy[-1])
            #Check data length
            if ( reload_df.shape[0] < (term_for_cal + day_for_buy[-1] + tech_indicator.have_to_cut_num) ):
                print("Shortage data / Code : {} / Shape : {}".format(code, reload_df.shape))
                continue

            #Preprocessing Data Set
            train_input, train_target, val_input, val_label = preprocessing.return_preprocessed_df(df=reload_df, profit_terms=day_for_buy, data_split_type='simulate')
            train_input = pd.DataFrame(train_input, columns=preprocessing.tot_col_names)
            val_input = pd.DataFrame(val_input, columns=preprocessing.tot_col_names)

            #For changing to categorical columns
            if (feature_cate_quantile_num!=0):
                train_input, val_input = quantile_df.return_quantile_df(df=train_input, quantile_num=feature_cate_quantile_num, valid_opt=True, valid_set=val_input)
                train_input = train_input.astype('category')
                val_input = val_input.astype('category')

            #For predicting day
            pred_dd_list=[]
            rmse_dd_list=[]
            real_dd_list=[]
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
                    
                fin_rate_tot = 0
                fin_rate_rmse = 0
                fin_rate_target = 0
                fin_rate_num = 0
                for mod_op in model_opts:

                    fin_rate, rmse_, real_target = model_part.get_result_from_model(train_input=input_for_train_in_idx, train_target=target_for_feature_selection,\
                        running_opt='simulate', val_input=input_for_valid_in_idx, val_target=val_label[idx], model_opt=mod_op,\
                        feature_cate_quantile_num=feature_cate_quantile_num, selected_col=selected_col_list)

                    fin_rate_tot += fin_rate
                    fin_rate_rmse += rmse_
                    fin_rate_target = real_target
                    fin_rate_num += 1
                pred_dd_list.append( fin_rate_tot / fin_rate_num )
                real_dd_list.append( fin_rate_target )
                rmse_dd_list.append( fin_rate_rmse / fin_rate_num )
            
            pred_dd_arr = np.array(pred_dd_list)
            real_dd_arr = np.array(real_dd_list)
            rmse_dd_arr = np.array(rmse_dd_list)

            """
            if (pred_dd_arr.mean()>=1.01):
                order_list.append(code)
                pred_list.append(pred_dd_arr-rmse_dd_arr)
                real_list.append(real_dd_arr)
                print("Code : {} / Prediction : {} / Real : {} / RMSE : {}".format(code, pred_dd_arr, real_dd_arr, rmse_dd_arr))
            else:
                print("Code {} is Not enough rate....".format(code))

            """
            order_list.append(code)
            pred_list.append(pred_dd_arr-rmse_dd_arr)
            real_list.append(real_dd_arr)
            print("Code : {} / Prediction : {} / Real : {} / RMSE : {}".format(code, pred_dd_arr, real_dd_arr, rmse_dd_arr))


        ###### Sorting...
        order_arr = np.array(order_list)
        pred_arr = np.array(pred_list)
        real_arr = np.array(real_list)

        pred_index = pred_arr.mean(axis=1).argsort()
        total_real_arr_mean = real_arr.mean(axis=0)
        max_real_profit_rate = real_arr[real_arr.mean(axis=1).argsort()][-5:].mean(axis=0)

        order_arr_selected = order_arr[pred_index][-5:]
        pred_arr_selected = pred_arr[pred_index][-5:]
        real_arr_selected = real_arr[pred_index][-5:]
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Total result for stat @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        for day_idx in range(len(day_for_buy)):
            print("Day : {} / 1.01 up : {}".format(day_for_buy[day_idx], real_arr[real_arr[:,day_idx]>=1.01].mean(axis=0)))
            print("Day : {} / 1.01 Down : {}".format(day_for_buy[day_idx], real_arr[real_arr[:,day_idx]<1.01].mean(axis=0)))
            print("Day : {} / 0.98 Down : {}".format(day_for_buy[day_idx], real_arr[real_arr[:,day_idx]<=0.98].mean(axis=0)))

        print("Last Day / 1.03 Up : {}".format(real_arr[real_arr[:,-1]>=1.03].mean(axis=0)))
        print("Last Day / 1.03 Down : {}".format(real_arr[real_arr[:,-1]<1.03].mean(axis=0)))
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if (len(order_arr)>=5):
            #Input to book
            order_book[trading_date_int] = order_arr_selected
            pred_book[trading_date_int] = pred_arr_selected
            real_book[trading_date_int] = real_arr_selected

            #Pringting Result
            cum_day+=1
            print("##################################### Order Book Result ############################################")
            print("Selling day : {} / Trading cum day : {}".format(trading_date_int, cum_day))
            print(order_arr_selected)
            print(pred_arr_selected)
            print(real_arr_selected)
            real_sale_rate=[]
            for rea in real_arr_selected:
                stop_whe=False
                for re in rea:
                    if ((re>=1.02)|(re<=0.98)):
                        real_sale_rate.append(re)
                        stop_whe=True
                        break
                if stop_whe:
                    stop_whe=False
                else:
                    real_sale_rate.append(rea[-1])
                    stop_whe=False
            real_sale_rate = np.array(real_sale_rate)
            print("Mean Profit : {}".format(real_sale_rate.mean()))
            print("Sale rate : {}".format(real_sale_rate))
            print("Real Profit rate : {}".format(real_sale_rate.mean()-0.003))
            print("Total profit rate mean: {}".format(total_real_arr_mean))
            print("Max profit....")
            print(max_real_profit_rate)
            origin_money = origin_money * real_sale_rate
            cum_money = origin_money.sum()
            print("Now money from stock : {} / Total sum : {}".format(origin_money, cum_money))
            origin_money = np.full(5, cum_money/5.0)
            print("#################################################################################")
            print()
            trading_day += timedelta(days=day_for_buy[-1]+1)
        else:
            print("Not enough to buy....")
            trading_day += timedelta(days=1)

    return order_book, pred_book, real_book

