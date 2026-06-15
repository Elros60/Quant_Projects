import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

from sklearn import linear_model
import lightgbm as lgbm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

import shap

import core.data_loader as data_loader
from core.feature_generator import FeatureGenerator
'''
This is the main file of the project AlphaAgent, which will serve as a toy-model for a multi-layer construction of alpha signal search engine.
The main idea contains the following layers:
- Data layer: this layer will be responsible for loading and preprocessing the data, including feature engineering and selection.
- Feature layer: this layer will be responsible for creating and selecting features for the models.
- Alpha layer: this will generate alpha signals based on the features.
- Model layer: this layer will be responsible for training and evaluating the models.
- LLM-layer: this layer will be responsible for generating insights and explanations based on the models and alpha signals.
- Graph-layer: this layer will construct a metric-graph of alpha signals.
'''

def main():
    #### Load input data, specifically the S&P 500 index data from Yahoo Finance
    # S&P 500 data will be less noisy than individual stock data, and will provide a good benchmark for the toy model as starting point.
    # ETFs like SPY or VOO can also be used as alternatives to the S&P 500 index data.
    # For alternative data sources, consider using Quandl or Alpha Vantage APIs, which provide a wide range of financial data.
    data = data_loader.load_yahoo_data('SPY', start='2000-01-01', end='2004-01-01')
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    feature_generator = FeatureGenerator()
    data = feature_generator.generate(data)
    data.to_csv('data/processed_data.csv', index=True)

    # IC (Spearman) check for features
    features = data.columns.drop(['Close', 'High', 'Low', 'Open', 'Volume', 'target'])
    # Rolling IC calculation
    window = 252  # 1 year of trading days
    IC_stats = []
    for feature in features:
        # Rolling IC
        rolling_ic = data[feature].rolling(window).corr(data['ret_1'].shift(-1))
        plt.figure(figsize=(12, 4))
        rolling_ic.plot()
        plt.title(f'Rolling IC for features (window={window})')
        plt.ylabel('IC (Pearson Correlation)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.savefig(f'data/rollingIC/rolling_ic_{feature}.png')
        plt.close()  # Close the plot to free up memory

        # Summary stats
        IC_stats.append({
            "feature": feature,
            "mean_ic": rolling_ic.mean(),
            "std_ic": rolling_ic.std(),
            "min_ic": rolling_ic.min(),
            "max_ic": rolling_ic.max()
        })
        
    IC_stats_df = pd.DataFrame(IC_stats)
    IC_stats_df.to_csv("data/feature_ic_stats.csv", index=False)
        

if __name__ == '__main__':
    main()