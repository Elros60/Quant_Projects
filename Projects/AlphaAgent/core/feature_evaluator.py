import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def calculate_rolling_tstat(rolling_ic, window):
    """
    Calculates the t-statistic for rolling IC values.

    t = r * sqrt(n-2) / sqrt(1 - r^2)
    """
    t_stat = rolling_ic * np.sqrt(window - 2) / np.sqrt(1 - rolling_ic**2)
    return t_stat


def momentum_regime(data, feature, target, horizon=1, window_short=50, window_long=200):
    """
    Determines the momentum regime based on moving averages.

    Parameters:
    - data: DataFrame containing the 'ret_1' column.
    - feature_rank: Ranked feature values.
    - target: Target variable.
    - horizon: The horizon for the target variable (default is 1).
    - window_short: The short-term moving average window (default is 50).
    - window_long: The long-term moving average window (default is 200).

    Returns:
    - low_momentum: Boolean Series indicating low momentum regime.
    - high_momentum: Boolean Series indicating high momentum regime.
    """
    ma_short = data[f"ret_{horizon}"].rolling(window_short).mean()
    ma_long = data[f"ret_{horizon}"].rolling(window_long).mean()
    threshold = ma_short - ma_long
    low_momentum = threshold <= 0
    high_momentum = threshold > 0

    # Momentum regime ICs
    ic_low_momentum = feature[low_momentum].corr(
        target[low_momentum], method="spearman"
    )
    ic_high_momentum = feature[high_momentum].corr(
        target[high_momentum], method="spearman"
    )

    return ic_low_momentum, ic_high_momentum


def volatility_regime(data, feature, target, horizon=1, window=63):
    """
    Determines the volatility regime based on realized volatility.

    Parameters:
    - data: DataFrame containing the 'ret_1' column.
    - feature_rank: Ranked feature values.
    - target: Target variable.
    - horizon: The horizon for the target variable (default is 1).
    - window: The rolling window size for volatility calculation (default is 63).

    Returns:
    - low_vol: Boolean Series indicating low volatility regime.
    - high_vol: Boolean Series indicating high volatility regime.
    """
    vol = data[f"ret_{horizon}"].rolling(window).std()
    threshold = vol.quantile(0.5)
    low_vol = vol <= threshold
    high_vol = vol > threshold

    # Volatility regime ICs
    ic_low_vol = feature[low_vol].corr(target[low_vol], method="spearman")
    ic_high_vol = feature[high_vol].corr(target[high_vol], method="spearman")

    return ic_low_vol, ic_high_vol


def sign_consistency(rolling_ic):
    """
    Evaluates the sign consistency of rolling IC values.

    Parameters:
    - rolling_ic: Series of rolling IC values.

    Returns:
    - sign_ratio: Fraction of windows matching overall direction.
    - flip_rate: Fraction of transitions where sign changes.
    - mean_run_length: Average length of consecutive same-sign runs.
    """
    sign = np.sign(rolling_ic).dropna()

    if sign.empty:
        return 0.0, 0.0, 0.0

    # Sign ratio: fraction of windows matching overall direction
    if rolling_ic.mean() > 0:
        sign_ratio = (sign > 0).mean()
    else:
        sign_ratio = (sign < 0).mean()

    # Flip rate: fraction of transitions where sign changes
    flip_rate = sign.diff().ne(0).sum() / max(1, (sign.shape[0] - 1))

    # Mean run length: average length of consecutive same-sign runs
    groups = (sign != sign.shift()).cumsum()
    run_lengths = sign.groupby(groups).size()
    mean_run_length = run_lengths.mean()

    return sign_ratio, flip_rate, mean_run_length


def plot_rolling_ic(rolling_ic, feature_name, horizon):
    """
    Plots the rolling Information Coefficient (IC) for a given feature.

    Parameters:
    - rolling_ic: Series of rolling IC values.
    - feature_name: Name of the feature being evaluated.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_ic.index, rolling_ic, label=f"Rolling IC: {feature_name}")
    plt.axhline(0, color="red", linestyle="--", label="Zero Line")
    plt.title(f"Rolling Information Coefficient (IC) for {feature_name}")
    plt.xlabel("Date")
    plt.ylabel("Rolling IC")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"data/figures/rolling_ic_{feature_name}_{horizon}.png")
    plt.close()


def feature_metrics(data, features, horizons, window=252, save_csv=True, plot=False):
    """
    Evaluates the rolling Information Coefficient (IC) and t-statistic for the given features.

    Parameters:
    - data: DataFrame containing the features and target variable.
    - features: List of feature names to evaluate.
    - horizons: List of horizons for the target variable.
    - window: Rolling window size for IC calculation (default is 252 trading days).
    - plot: Boolean indicating whether to plot the results (default is False).
    """
    feature_metrics = []

    for feature in features:
        for horizon in horizons:

            target = data[f"ret_{horizon}"].shift(-horizon)
            target_rank = target.rank(pct=True)

            # Rank the feature
            feature_series = data[feature]
            feature_rank = feature_series.rank(pct=True)

            # Overall IC (non-rolling, over entire dataset)
            overall_ic = feature_series.corr(target, method="spearman")

            # Calculate rolling IC
            rolling_ic = feature_rank.rolling(window).corr(target_rank)
            # rolling_tstat = calculate_rolling_tstat(rolling_ic, window)

            # Sign consistency
            sign_ratio, flip_rate, mean_run_length = sign_consistency(rolling_ic)

            # Regime analysis
            ic_low_momentum, ic_high_momentum = momentum_regime(
                data, feature_series, target, horizon=horizon
            )
            ic_low_vol, ic_high_vol = volatility_regime(
                data, feature_series, target, horizon=horizon
            )

            # Plotting
            if plot:
                plot_rolling_ic(rolling_ic, feature, horizon)

            # Summary stats
            feature_metrics.append(
                {
                    "feature": feature,
                    "horizon": horizon,
                    "overall_ic": overall_ic,
                    "sign_ratio": sign_ratio,
                    "flip_rate": flip_rate,
                    "mean_run_length": mean_run_length,
                    "mean_ic": rolling_ic.mean(),
                    "std_ic": rolling_ic.std(),
                    "min_ic": rolling_ic.min(),
                    "max_ic": rolling_ic.max(),
                    "icir": (
                        rolling_ic.mean() / rolling_ic.std()
                        if rolling_ic.std() != 0
                        else None
                    ),
                    # "mean_tstat": rolling_tstat.mean(),
                    # "std_tstat": rolling_tstat.std(),
                    # "min_tstat": rolling_tstat.min(),
                    # "max_tstat": rolling_tstat.max(),
                    "ic_low_momentum": ic_low_momentum,
                    "ic_high_momentum": ic_high_momentum,
                    "ic_low_vol": ic_low_vol,
                    "ic_high_vol": ic_high_vol,
                }
            )

    # Evaluate feature importance based on statistics
    feature_metrics_df = pd.DataFrame(feature_metrics)
    metrics_scoring(feature_metrics_df)  # Get importance score for each feature

    # Save summary stats to CSV
    if save_csv:
        feature_metrics_df.to_csv(
            "data/feature_metrics_horizon.csv", index=False, float_format="%.5f"
        )

    return feature_metrics_df


def check_feature_correlation(data, features, threshold=0.8):
    """
    Checks correlation between features to identify redundancy.

    Parameters:
    - data: DataFrame with features
    - features: List of feature names
    - threshold: Correlation threshold for redundancy (default 0.8)
    - plot: Boolean indicating whether to plot the correlation matrix (default True)

    Returns:
    - corr_matrix: Feature correlation matrix
    - redundant_pairs: List of (feature1, feature2, correlation) tuples
    """
    # Compute correlation matrix
    feature_data = data[features]
    corr_matrix = feature_data.corr()

    # TODO: might need to use IC-based correlation (correlations between features relative to that horizon's target)
    # To be more horizon-specific
    # Keep the current correlation between raw features for first run

    # Find redundant pairs
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                redundant_pairs.append((feat1, feat2, corr_val))

    # Sort by absolute correlation (highest first)
    redundant_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return corr_matrix, redundant_pairs


def first_level_feature_selection(
    feature_metrics_df,
    horizons,
    mean_ic_threshold=0.015,
    absmax_ic_threshold=0.05,
    overall_ic_threshold=0.02,
):
    """
    Performs first-level feature selection based on mean IC and max absolute IC across horizons.

    Parameters:
    - feature_metrics_df: DataFrame containing feature metrics
    - horizons: List of horizons for the target variable
    - mean_ic_threshold: Threshold for mean IC across horizons (default 0.015)
    - absmax_ic_threshold: Threshold for max absolute IC across horizons (default 0.05)
    - overall_ic_threshold: Threshold for overall IC (default 0.015)

    Returns:
    - selected_features: List of features that meet the selection criteria for each horizon
    """
    selected_features_list = []

    for horizon in horizons:
        byhorizon = feature_metrics_df[feature_metrics_df["horizon"] == horizon]
        selected_features = (
            byhorizon[
                (byhorizon["mean_ic"].abs() > mean_ic_threshold)
                & (byhorizon["max_ic"].abs() > absmax_ic_threshold)
                & (byhorizon["overall_ic"].abs() > overall_ic_threshold)
            ]
            .nlargest(10, "importance_score")["feature"]
            .tolist()
        )

        selected_features_list.append(
            {"horizon": horizon, "selected_features": selected_features}
        )

    selected_features_df = pd.DataFrame(selected_features_list)
    return selected_features_df


def remove_redundant_features(selected_features, summary, redundant_pairs, horizon):
    """
    Removes redundant features based on correlation and mean IC across horizons.

    Parameters:
    - selected_features: List of initially selected features
    - summary: DataFrame containing feature metrics summary
    - redundant_pairs: List of (feature1, feature2, correlation) tuples

    Returns:
    - selected_features: Updated list of features after removing redundancies
    """
    for pair in redundant_pairs:
        print(f"Redundant pair: {pair[0]} and {pair[1]} with correlation {pair[2]:.2f}")

        # Remove the feature with the lower mean IC across horizons
        if pair[0] not in selected_features or pair[1] not in selected_features:
            continue

        absmax_ic_0 = summary.loc[
            (summary["feature"] == pair[0]) & (summary["horizon"] == horizon),
            "importance_score",
        ].values[0]
        absmax_ic_1 = summary.loc[
            (summary["feature"] == pair[1]) & (summary["horizon"] == horizon),
            "importance_score",
        ].values[0]

        if absmax_ic_0 < absmax_ic_1:
            selected_features.remove(pair[0])
            print(f"Removed {pair[0]} due to lower importance score.")
        else:
            selected_features.remove(pair[1])
            print(f"Removed {pair[1]} due to lower importance score.")

    return selected_features


def metrics_scoring(IC_stats_df):
    """
    Computes a composite importance score for each feature based on various metrics.
    To-be finalized with multi-horizon analysis.
    """

    # Final importance score calculation
    IC_stats_df["importance_score"] = (
        abs(IC_stats_df["mean_ic"])
        * 0.3  # Mean IC is the most important metric, for predictive power
        + abs(IC_stats_df["overall_ic"])
        * 0.3  # Overall IC, which is the correlation over the entire dataset, is also important
        + abs(
            IC_stats_df["icir"]
        )  # ICIR is the ratio of mean IC to std IC, which measures the consistency of the feature's predictive power
        * 0.1  # Consistency of the feature's predictive power
        + IC_stats_df["sign_ratio"] * 0.1  # Sign consistency, temporal stability
        + 1
        / (1 + abs(IC_stats_df["ic_low_momentum"] - IC_stats_df["ic_high_momentum"]))
        * 0.05  # Regime consistency
        + 1
        / (1 + abs(IC_stats_df["ic_low_vol"] - IC_stats_df["ic_high_vol"]))
        * 0.05  # Regime consistency
        + np.sign(IC_stats_df["ic_low_momentum"])
        * np.sign(IC_stats_df["ic_high_momentum"])
        * 0.05  # Regime consistency
        + np.sign(IC_stats_df["ic_low_vol"])
        * np.sign(IC_stats_df["ic_high_vol"])
        * 0.05  # Regime consistency
        - IC_stats_df["flip_rate"]
        * 0.05  # Penalty for flip rate and regime inconsistency
    )
