from common import lookup, summary
import pandas as pd
import numpy as np

def reverse_list(list_data):
	new_list = []
	while list_data:
		new_list.append(list_data.pop())
	return new_list


def moment_data(stock_code, chart_type):
	result_dict={}
	num_of_data = 30

	lookup_class_ob = lookup.Lookup_past_data()
	moment_data = lookup_class_ob.get_data_for_present(stock_code=stock_code, num_of_data=num_of_data, chart_type=chart_type)

	if len(moment_data['close'])<30:

		result_dict["macd"] = -1
		result_dict["emacd"] = -1
		result_dict["rsi"] = -1
		result_dict["stoch_k"] = -1
		result_dict["mfi"] = -1

		result_dict["present_price"] = moment_data['present_price']
		result_dict["present_volume"] = moment_data['present_volume']
		result_dict["present_per"] = moment_data['present_per']
		result_dict["present_eps"] = moment_data['present_eps']
		result_dict["present_program_bought"] = moment_data['present_program_bought']
		result_dict["present_foreigner_bouht"] = moment_data['present_foreigner_bouht']
		result_dict["present_corp_bought"] = moment_data['present_corp_bought']
		return result_dict

	past_close_data = reverse_list(moment_data['close'])
	past_volume_data = reverse_list(moment_data['volume'])

	past_close_data_series = pd.Series(past_close_data)
	past_volume_data_series = pd.Series(past_volume_data)

	macd_series = summary.macd(past_close_data_series, 12, 26)
	emacd_series = summary.emacd(past_close_data_series, 12, 26)
	rsi_series = summary.rsi(past_close_data_series, 14)
	stoch_k_series = summary.stochastic_index(past_close_data_series, 14)
	mfi_series = summary.mfi(past_close_data_series, past_volume_data_series, 14)

	result_dict["macd"] = macd_series.values[-1]
	result_dict["emacd"] = emacd_series.values[-1]
	result_dict["rsi"] = rsi_series.values[-1]
	result_dict["stoch_k"] = stoch_k_series.values[-1]
	result_dict["mfi"] = mfi_series.values[-1]

	result_dict["present_price"] = moment_data['present_price']
	result_dict["present_volume"] = moment_data['present_volume']
	result_dict["present_per"] = moment_data['present_per']
	result_dict["present_eps"] = moment_data['present_eps']
	result_dict["present_program_bought"] = moment_data['present_program_bought']
	result_dict["present_foreigner_bouht"] = moment_data['present_foreigner_bouht']
	result_dict["present_corp_bought"] = moment_data['present_corp_bought']

	return result_dict