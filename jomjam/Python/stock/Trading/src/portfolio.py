from .utils import return_data
from .trend import check_trend

def filter_code(code_list, get_data_start_datetime, get_data_end_datetime):
    start_datetime_right_format = get_data_start_datetime.strftime("%Y-%m-%d")
    end_datetime_right_format = get_data_end_datetime.strftime("%Y-%m-%d")

    # 기준 값을 삼성으로 활용
    get_length = len(return_data("005930", start_datetime_right_format, end_datetime_right_format).Close)
    
    # Empty list 생성
    results = []

    for code in code_list:
        # 타겟 데이터 불러오기
        tmp_df = return_data(code, start_datetime_right_format, end_datetime_right_format)
        # 데이터 충분한지 확인 및 값 존재 여부 확인
        if tmp_df.shape[0] == get_length and "Close" in tmp_df.columns :

            ######################################
            # Trend check
            trend_idx = check_trend(arr=tmp_df.Close.values, window=10)
            if trend_idx > 0 :
                



            # Signal check
            # then append
            ######################################

    return results
    