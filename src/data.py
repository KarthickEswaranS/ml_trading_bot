import keyring
from binance import Client
import pandas as pd


class Data:

    def __init__(self):
        api_key = keyring.get_password('binance', 'api_key')
        api_secret = keyring.get_password('binance', 'api_secret')
        self.client = Client(api_key, api_secret)
        self.hist_data = self.client.get_historical_klines(
            'BTCUSDT',Client.KLINE_INTERVAL_1DAY,
            '1-1-2023',
            '1-1-2025',
        )

    def data_cleaning(self):
        df = pd.DataFrame(self.hist_data)
    
        df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close',
                       'Volume', 'Close Time','Quote Asset Volume',
                       'Number of Trades', 'Taker Buy Base Asset Volume',
                        'Taker Buy Quote Asset Volume', 'Ignore']
       
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit= 'ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit= 'ms')
        to_numeric_data = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
        df[to_numeric_data] = df[to_numeric_data].apply(pd.to_numeric)
    
        df.set_index('Open Time', inplace=True)
        return df
    

# c = Data()
# d = c.data_cleaning()
# f = c.feature(d)
# t = c.train_test(f)

# split_index = int(len(f) * 0.8)

# b = c.backtest(f,t, split_index)
