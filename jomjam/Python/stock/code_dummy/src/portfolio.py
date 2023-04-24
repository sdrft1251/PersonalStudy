import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from .utils import return_data, get_stock_code_list
from .method import return_reference_data

# 종료 지점 및 기간으로 가져오기
def make_comb(code_list, stand_datetime, during_time_delta):
    stand_datetime_right_format = stand_datetime.strftime("%Y-%m-%d")
    start_datetime_right_format = (stand_datetime - during_time_delta).strftime("%Y-%m-%d")

    # 삼성을 기준으로 길이 가져오기
    get_length = len(return_data("005930", start_datetime_right_format, stand_datetime_right_format).Close)
    close_colls = []
    used_code_list = []
    for code in code_list:
        tmp_df = return_data(code, start_datetime_right_format, stand_datetime_right_format)
        if tmp_df.shape[0] == get_length and "Close" in tmp_df.columns :
            close_colls.append(tmp_df.Close.values)
            used_code_list.append(code)

    close_colls_arr = np.array(close_colls)
    used_code_list = np.array(used_code_list)
    return close_colls_arr, used_code_list

# 시작 지점 및 종료 지점으로 가져오기
def make_comb(code_list, start_datetime, end_datetime):
    start_datetime_right_format = start_datetime.strftime("%Y-%m-%d")
    end_datetime_right_format = end_datetime.strftime("%Y-%m-%d")

    # 삼성을 기준으로 길이 가져오기
    get_length = len(return_data("005930", start_datetime_right_format, end_datetime_right_format).Close)
    close_colls = []
    used_code_list = []
    for code in code_list:
        tmp_df = return_data(code, start_datetime_right_format, end_datetime_right_format)
        if tmp_df.shape[0] == get_length and "Close" in tmp_df.columns :
            close_colls.append(tmp_df.Close.values)
            used_code_list.append(code)

    close_colls_arr = np.array(close_colls)
    used_code_list = np.array(used_code_list)
    return close_colls_arr, used_code_list


################################################################################################################################################################################################
# 포트폴리오화
################################################################################################################################################################################################
def select_code(df, method, hedging={"whether":False, "corr_stand_val":0.9, "op_num":1}, **kwars):
    # 참조 값 가져오기
    ref_code = return_reference_data(df=df, method=method, kwars)
    # 헷징
    if hedging["whether"]:
        ref_code, hedging_result_list = hedging(df=df, code_list=ref_code, corr_stand_val=hedging["corr_stand_val"], op_num=hedging["op_num"])
        return ref_code, hedging_result_list
    return ref_code, []


def return_portfolio(get_data_start_datetime, get_data_end_datetime, money_by_code, **portfolio_opt):
    # 초기 값 설정 및 확인
    hed_opt = False
    try:
        if ~portfolio_opt["method"]:
            raise Exception("포트폴리오 구성 Method 입력 필요")
        if ~portfolio_opt["detail"]:
            raise Exception("포트폴리오 구성에 대한 Detail 필요 (Method 마다 있음)")
    except Exception as e:
        print('Error! : ', e)
        print("형식 : portfolio_opt = {'method': SOMETHING, 'hedging': SOMETHING, 'detail': SOMETHING_DICT}")
    if ~portfolio_opt["hedging"]:
        portfolio_opt["hedging"] = {"whether":False, "corr_stand_val":0.9, "op_num":1}
    
    # 기초 데이터 프레임 만들기
    code_list = list(get_stock_code_list("KOSPI").Symbol.values)
    new_code_list = []
    for code in code_list:
        if code.isdigit():
            new_code_list.append(code)
    result_price, result_code = make_comb(code_list=new_code_list, start_datetime=get_data_start_datetime, end_datetime = get_data_end_datetime)
    print("Past price data shape is : {}".format(result_price.shape))
    price_df = pd.DataFrame(result_price.T, columns = result_code)
    # null 제거를 위한
    delete_col_list = []
    for col in price_df.columns:
        ser = price_df[col]
        if ser.max() != ser.min():
            delete_col_list.append(col)
    dropped_price_df = price_df.drop(delete_col_list, axis=1)
    print("Delete Null columns = Before Shape : {} /  After Shape : {}".format(price_df.shape, dropped_price_df.shape))
    # 종목 가져오기
    ref_code, hedging_result_list = select_code(df=dropped_price_df, method=portfolio_opt["method"], hedging=portfolio_opt["hedging"], portfolio_opt["detail"])
    print("종목 선정 결과 = 총 개수 : {}".format(len(ref_code)))
    # 헷징 확인
    if len(hedging_result_list)>=1:
        hed_opt = True
    code_list = []
    retain_list = []
    test_start_datetime = get_data_end_datetime
    for idx, code in enumerate(ref_code):
        # 해당 날짜에 데이터가 없다면, 하루 씩 추가해서 다시 살펴보기
        while True:
            try:
                return_data(code, test_start_datetime.strftime("%Y-%m-%d"), test_start_datetime.strftime("%Y-%m-%d")).Close.values[0]
                break
            except:
                test_start_datetime = test_start_datetime + timedelta(days=1)
        # 개수 추출
        data = return_data(code, test_start_datetime.strftime("%Y-%m-%d"), test_start_datetime.strftime("%Y-%m-%d")).Close.values[0]
        can_buy_num = int(money_by_code/data)
        # 헷징 추가 부분
        if hed_opt:
            data_hedge = return_data(hedging_result_list[idx], test_start_datetime.strftime("%Y-%m-%d"), test_start_datetime.strftime("%Y-%m-%d")).Close.values[0]
            can_buy_num_hedge = int(money_by_code/data_hedge)
            if can_buy_num>=1 and can_buy_num_hedge>=1:
                code_list.append(code)
                code_list.append(hedging_result_list[idx])
                retain_list.append(data*can_buy_num)
                retain_list.append(data_hedge*can_buy_num_hedge)
        else:
            if can_buy_num>=1:
                code_list.append(code)
                retain_list.append(data*can_buy_num)
        
    print("Porfolio Result = 총 개수 : {}".format(len(code_list)))
    return code_list, retain_list

