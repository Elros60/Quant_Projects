import numpy as np
import pandas as pd
'''
This file contains the feature generation functions.
'''
class FeatureGenerator:

    def __init__(self, momentum_periods=[3,5,10,15,20], vol_periods=[20,60,120]):
        self.momentum_periods = momentum_periods
        self.vol_periods = vol_periods

    def generate(self, df):

        df = df.copy()
        df["ret_1"] = np.log(df["Close"]).diff()

        for i in self.momentum_periods:

            # Momentum
            df[f"ret_{i}"] = df["Close"].pct_change(i)

            # Exponential Moving Average (EMA)
            hl = i / 2
            alpha = np.log(2)/hl
            ewm_weights = np.exp(-alpha*np.arange(i))
            ewm_weights /= ewm_weights.sum()
            df[f"ema_{i}"] = df["Close"].rolling(i).apply(lambda x: np.sum(ewm_weights*x), raw=True)


        # Moving average 20 days
        ma20 = df["Close"].rolling(20).mean()
        df["price_ma20_ratio"] = (
            df["Close"] / ma20
        )

        for i in self.vol_periods:
            # Volatility
            df[f"vol_{i}"] = (
                df["ret_1"]
                .rolling(i)
                .std()
            )

            # Volume
            mean = df["Volume"].rolling(i).mean()
            std = df["Volume"].rolling(i).std()

            df[f"volume_ratio_{i}"] = (
                df["Volume"]
                / df["Volume"].rolling(i).mean()
            )

            df[f"volume_zscore_{i}"] = (
                (df["Volume"] - mean) / std
            )

        df["target"] = (df["ret_1"].shift(-1) > 0).astype(int)
        df = df.dropna()

        return df