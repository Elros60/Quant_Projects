import yfinance as yf
import pandas as pd
'''
This file contains the data loading and related data processing functions.
'''
def load_yahoo_data(ticker, start, end, interval='1d'):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    df.index = pd.to_datetime(df.index)
    return df