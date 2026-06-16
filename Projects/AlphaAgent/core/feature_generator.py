import numpy as np
import pandas as pd
"""
This file contains the feature generation functions.
"""
class FeatureGenerator:
    """
    This class is responsible for generating features from the input data.
    It takes a DataFrame as input and returns a DataFrame with the generated features.
    The features generated include:
    - Momentum features: ret_3, ret_5, ret_10, ret_15, ret_20
    - Exponential Moving Average (EMA) features: ema_3, ema_5, ema_10, ema_15, ema_20
    - Moving average 20 days feature: price_ma20_ratio
    - Volatility features: vol_20, vol_60, vol_120
    - Volume features: volume_ratio_20, volume_zscore_20, volume_ratio_60, volume_zscore_60, volume_ratio_120
    - Target variable: target (1 if ret_1 > 0, 0 otherwise)
    """
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