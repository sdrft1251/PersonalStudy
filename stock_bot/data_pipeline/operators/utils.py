import FinanceDataReader as fdr
from datetime import datetime, timedelta



def get_stock_past_data(code, start_date=None, end_date=None):
    """
    Columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")

    data = fdr.DataReader(code, start_date, end_date)

    return data


def get_market_list(symbol):
    """
    Columns = ['Symbol', 'Market', 'Name', 'Sector', 'Industry', 'ListingDate',
                'SettleMonth', 'Representative', 'HomePage', 'Region']
    """
    return fdr.StockListing(symbol)


def get_code_list(symbol):
    data = get_market_list(symbol)
    return data['Symbol'].values


def replace_datetime_to_string(raw):
    return raw.strftime("%Y-%m-%d")