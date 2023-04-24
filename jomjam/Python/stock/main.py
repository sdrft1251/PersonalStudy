import utils

stocks_df = utils.get_stock_code_list('NASDAQ')
print(stocks_df.shape[0])
outs_nasdaq_2020 = utils.find_top_market_cap(stocks_df, 2020, 50)
print(outs_nasdaq_2020)
outs_nasdaq_2010 = utils.find_top_market_cap(stocks_df, 2010, 50)
print(outs_nasdaq_2010)
outs_nasdaq_2000 = utils.find_top_market_cap(stocks_df, 2000, 50)
print(outs_nasdaq_2000)
