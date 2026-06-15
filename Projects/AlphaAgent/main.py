import pandas as pd
import numpy as np
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
    data = data_loader.load_yahoo_data('SPY', start='2000-01-01', end='2001-01-01')
    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    feature_generator = FeatureGenerator()
    data = feature_generator.generate(data)
    data.to_csv('data/processed_data.csv', index=True)

    # IC (Spearman) check for features
    features = data.columns.drop(['target'])
    for feature in features:
        ic = data[feature].corr(data['ret_1'].shift(-1), method='spearman')
        print(f"Feature: {feature}, IC: {ic:.4f}")

if __name__ == '__main__':
    main()