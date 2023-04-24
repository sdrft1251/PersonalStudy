import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import return_data

################################################################
################### 포트폴리오 유지 전략을 위한 코드 ###################
################################################################

# 기준일 가격반환 모듈
def cal_present_rate(portfolio_info, datenow):
    present_price_arr = np.zeros(portfolio_info.shape[0])
    for idx, code in enumerate(portfolio_info["code"]):
        present_price_arr[idx] = return_data(code=code, start_date=datenow, end_date=datenow).Close
    return present_price_arr

# 데이터 프레임에 신규 날짜의 비중 추가
def cal_diff_between_two_day(portfolio_info, before_date, after_date, stand_rate, before_balance):
    # Before Trade
    before_price_arr = cal_present_rate(portfolio_info, before_date)
    after_price_arr = cal_present_rate(portfolio_info, after_date)
    
    before_retain = portfolio_info[before_date].values
    increase_rate = after_price_arr / before_price_arr
    after_retain = before_retain * increase_rate

    after_retain_rate = after_retain / after_retain.sum()
    rate_diff = after_retain_rate - stand_rate

    # Decide Trade
    # 비율도 변화하였고, 오르기도한 부분만 해야하지 않을까?

    # 비중 증가 부분 -> 비중 증가분 만큼 팔기
    increased_mask = np.where(increase_rate>1, 1, 0)
    can_sale = after_retain.sum() * (increased_mask * np.where(rate_diff > 0, rate_diff, 0))
    sale_stock_num = (can_sale/after_price_arr).astype("int")
    surplus = sale_stock_num * after_price_arr
    after_sale_retain = after_retain - surplus
    # 수수료 제외 시뮬레이션
    surplus_sum_except_fees = surplus.sum() * 0.995


    # 잉여돈으로 낮은 비중 매입
    decreased_mask = np.where(increase_rate<1, 1, 0)
    can_buy = after_retain.sum() * (decreased_mask * np.where(rate_diff<0, -rate_diff, 0))
    if can_buy.sum() == 0:
        buy_stock_num = np.zeros(len(after_price_arr))
    else: 
        can_buy = (surplus_sum_except_fees + before_balance) * (can_buy / can_buy.sum())
        buy_stock_num = (can_buy/after_price_arr).astype("int")
    add_stock = buy_stock_num * after_price_arr
    after_buy_retain = after_sale_retain + add_stock

    balanced = surplus_sum_except_fees + before_balance - add_stock.sum()
    if balanced<0:
        print("ERROR!!")
    print("Surplus : {} / BuyAmount : {} / Balance : {} / Sum : {}".format(surplus_sum_except_fees, add_stock.sum(), balanced, after_buy_retain.sum()))
    print(sale_stock_num, buy_stock_num)

    # Adding
    portfolio_info[after_date] = after_buy_retain
    return portfolio_info, balanced

def cycle_module(portfolio_info, before_list, after_list):
    start_retain_rate = portfolio_info[before_list[0]].values / portfolio_info[before_list[0]].values.sum()
    balanced = 0
    for idx, val in enumerate(before_list):
        portfolio_info, balanced = cal_diff_between_two_day(portfolio_info, val, after_list[idx], start_retain_rate, balanced)
    return portfolio_info, balanced

def test_start_func(code_list, retain_list, test_start_datetime, test_end_datetime):
    for_date_df = return_data("005930", test_start_datetime.strftime("%Y-%m-%d"), test_end_datetime.strftime("%Y-%m-%d"))
    before_list = for_date_df.index[:-1]
    after_list = for_date_df.index[1:]

    port_ex = pd.DataFrame()
    port_ex['code'] = code_list
    port_ex[before_list[0]] = retain_list
    print("Back Testing Start -----------------------")
    result_df, balance = cycle_module(port_ex, before_list, after_list)
    print("Result is -----------------------")
    val = (result_df.sum(axis=0).iloc[1:]).astype(np.float32)
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(val)), val.values)
    plt.show()
    print("Balance & Profit rate is {} / {}".format(balance, (val[-1] - val[0] + balance) / val[0]))
    return 1 + ((val[-1] - val[0] + balance) / val[0])




