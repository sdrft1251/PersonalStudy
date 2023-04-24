import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
#########################################################
############ 포트폴리오 구성 방법 모듈
#########################################################

#### 헷징 ####
def hedging(df, code_list, corr_stand_val=0.9, op_num=1):
    code_result_list = []
    hedging_result_list = []
    # 상관 데이터 프레임 생성
    corr_df = df.corr()
    for code in code_list:
        corr_ser = corr_df[key].sort_values()
        cand_code = corr_ser[corr_ser < corr_stand_val]
        if len(cand_code)>=op_num:
            code_result_list.append(code)
            hedging_result_list.append(cand_code.index[:op_num])
    return code_result_list, hedging_result_list

#### 전략 ####
def return_reference_data(df, method, **kwargs):

    if method == "simple_increase" :
        print("단순 증가")
        # 값 선언
        tot_length = df.shape[0]
        low_increase_rate = kwargs["low_increase_rate"]
        high_increase_rate = kwargs["high_increase_rate"]
        extract_num = kwargs["extract_num"]
        # 함수 시작
        coef_dict = {}
        for code in df.columns:
            # 가격 데이터 -> 정규화 시키기
            price_by_code = df[code].values
            normed_price_by_code = price_by_code / price_by_code[0]
            # 선형에 적합
            line_fitter = LinearRegression()
            line_fitter.fit(np.arange(len(normed_price_by_code)).reshape(-1,1), normed_price_by_code)
            # 기울기 값 추가
            coef_val = line_fitter.coef_[0]
            if coef_val>=low_increase_rate and coef_val<=high_increase_rate:
                coef_dict[code] = coef_val
        # 기울기 값으로 정렬
        sorted_coef_dict = sorted(coef_dict.items(), key=(lambda x:x[1]), reverse=True)
        # 원하는 개수 추출
        result = sorted_coef_dict[:extract_num]
        # 코드만 걸러내어 Return
        return np.array(result)[:,0]

    elif method == "simple_increase_and_low_shortma" :
        print("단순 증가 및 낮은 단지 MA")
        # 값 선언
        tot_length = df.shape[0]
        long_length = int(tot_length*kwargs["long_rate"])
        short_length = int(tot_length*kwargs["short_rate"])
        low_increase_rate = kwargs["low_increase_rate"]
        high_increase_rate = kwargs["high_increase_rate"]
        extract_num = kwargs["extract_num"]
        # 함수 시작
        for_sort_dict = {}
        for code in df.columns:
            # 가격 데이터 -> 정규화 시키기
            price_by_code = df[code].values
            normed_price_by_code = price_by_code / price_by_code[0]
            # 선형에 적합
            line_fitter = LinearRegression()
            line_fitter.fit(np.arange(len(normed_price_by_code)).reshape(-1,1), normed_price_by_code)
            coef_val = line_fitter.coef_[0]
            # 데이터 저장
            if coef_val>=low_increase_rate and coef_val<=high_increase_rate:
                # 단기 및 장기 MA 값 추출
                long_ma = price_by_code[-long_length:].mean()
                short_ma = price_by_code[-short_length:].mean()
                # 단기와 장기의 차이를 비중으로
                suff_ma = (long_ma-short_ma)/long_ma
                for_sort_dict[code] = suff_ma
        # 값 정렬
        sorted_dict = sorted(for_sort_dict.items(), key=(lambda x:x[1]), reverse=True)
        # 원하는 개수 추출
        result = sorted_coef_dict[:extract_num]
        # 코드만 걸러내어 Return
        return np.array(result)[:,0]
                




    return -1