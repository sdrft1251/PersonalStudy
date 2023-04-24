import win32com.client
import pythoncom



class Lookup_past_data:
    def __init__(self):
        super().__init__()


    def get_data_for_present(self, stock_code, num_of_data, chart_type):

        ##### Market Eye part #####
        pythoncom.CoInitialize()
        instMarketEye = win32com.client.Dispatch("CpSysDib.MarketEye")

        stock_code_for_cp = "A" + str(stock_code)
        for_return = {'close':[], 'volume':[]}

        # Get present data first
        instMarketEye.SetInputValue(0, (4, 10, 67, 70, 116, 118, 120))
        instMarketEye.SetInputValue(1, stock_code_for_cp)

        instMarketEye.BlockRequest()
        for_return['present_price'] = instMarketEye.GetDataValue(0, 0)
        for_return['present_volume'] = instMarketEye.GetDataValue(1, 0)
        for_return['present_per'] = instMarketEye.GetDataValue(2, 0)
        for_return['present_eps'] = instMarketEye.GetDataValue(3, 0)
        for_return['present_program_bought'] = instMarketEye.GetDataValue(4, 0)
        for_return['present_foreigner_bouht'] = instMarketEye.GetDataValue(5, 0)
        for_return['present_corp_bought'] = instMarketEye.GetDataValue(6, 0)

        #pythoncom.CoUninitialize()
        ##### END #####

        ##### Stock Chart part #####
        #pythoncom.CoInitialize()
        instStockChart = win32com.client.Dispatch("CpSysDib.StockChart")

        # Get Past Data
        instStockChart.SetInputValue(0, stock_code_for_cp)
        instStockChart.SetInputValue(1, ord('2'))
        instStockChart.SetInputValue(4, num_of_data)
        instStockChart.SetInputValue(5, (5, 8))
        instStockChart.SetInputValue(6, ord(chart_type))
        instStockChart.SetInputValue(9, ord('1'))

        instStockChart.BlockRequest()

        numData = instStockChart.GetHeaderValue(3)
        numField = instStockChart.GetHeaderValue(1)

        for i in range(numData):
            for_return['close'].append(instStockChart.GetDataValue(0, i))
            for_return['volume'].append(instStockChart.GetDataValue(1, i))


        pythoncom.CoUninitialize()
        ##### END #####
        print("Process END!!!!")
        return for_return