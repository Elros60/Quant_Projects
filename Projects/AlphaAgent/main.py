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
import core.feature_evaluator as feature_evaluator
from core.feature_generator import FeatureGenerator

"""
This is the main file of the project AlphaAgent, which will serve as a toy-model for a multi-layer construction of alpha signal search engine.
The main idea contains the following layers:
- Data layer: this layer will be responsible for loading and preprocessing the data, including feature engineering and selection.
- Feature layer: this layer will be responsible for creating and selecting features for the models.
- Alpha layer: this will generate alpha signals based on the features.
- Model layer: this layer will be responsible for training and evaluating the models.
- LLM-layer: this layer will be responsible for generating insights and explanations based on the models and alpha signals.
- Graph-layer: this layer will construct a metric-graph of alpha signals.
"""


def main():
    #### Load input data, specifically the S&P 500 index data from Yahoo Finance

    # S&P 500 data will be less noisy than individual stock data, and will provide a good benchmark for the toy model as starting point.
    # ETFs like SPY or VOO can also be used as alternatives to the S&P 500 index data.
    # For alternative data sources, consider using Quandl or Alpha Vantage APIs, which provide a wide range of financial data.
    data = data_loader.load_yahoo_data("SPY", start="2000-01-01", end="2004-01-01")
    data.columns = ["Close", "High", "Low", "Open", "Volume"]

    # Generate features using the FeatureGenerator class
    horizons = [1, 5, 10, 20]
    momentum_periods = [1, 3, 5, 10, 15, 20]
    vol_periods = [20, 60, 120]

    mean_ic_threshold = 0.02
    absmax_ic_threshold = 0.03
    overall_ic_threshold = 0.015

    feature_generator = FeatureGenerator(momentum_periods, vol_periods)
    data = feature_generator.generate(data)

    # Save the processed data to a CSV file for further analysis and modeling
    data.to_csv("data/processed_data.csv", index=True)

    # Feature metrics evaluation
    features = data.columns.drop(["Close", "High", "Low", "Open", "Volume"])
    feature_metrics_df = feature_evaluator.feature_metrics(
        data, features, horizons, 63, True, True
    )

    # First level feature selection based on the highest absolute IC values for each feature across different horizons
    selected_features_per_horizon = feature_evaluator.first_level_feature_selection(
        feature_metrics_df,
        horizons=horizons,
        mean_ic_threshold=mean_ic_threshold,
        absmax_ic_threshold=absmax_ic_threshold,
        overall_ic_threshold=overall_ic_threshold,
    )

    selected_features_df = pd.DataFrame(selected_features_per_horizon)
    selected_features_df.to_csv(
        "data/feature_metrics_summary.csv", index=False, float_format="%.5f"
    )

    # Check feature correlation and identify redundant features
    selected_features_nocorr = []
    for row in selected_features_df.itertuples(index=False):

        print(f"Selected features for horizon {row.horizon}: {row.selected_features}")
        corr_matrix, redundant_pairs = feature_evaluator.check_feature_correlation(
            data, row.selected_features, threshold=0.9
        )

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", center=0)
        plt.title(f"Feature Correlation Matrix for horizon {row.horizon}")
        plt.tight_layout()
        plt.savefig(f"data/figures/feature_correlation_heatmap_{row.horizon}.png")
        plt.close()

        selected_features = feature_evaluator.remove_redundant_features(
            row.selected_features, feature_metrics_df, redundant_pairs, row.horizon
        )

        selected_features_nocorr.append(
            {"horizon": row.horizon, "selected_features": selected_features}
        )
        print(f"Selected features after correlation check: {selected_features}")

    selected_features_nocorr_df = pd.DataFrame(selected_features_nocorr)
    selected_features_nocorr_df.to_csv(
        "data/feature_metrics_summary_nocorr.csv", index=False, float_format="%.5f"
    )


if __name__ == "__main__":
    main()
