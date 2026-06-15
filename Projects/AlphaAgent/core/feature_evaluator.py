import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_rolling_tstat(rolling_ic, window):
    """
    Calculates the t-statistic for rolling IC values.
    
    t = r * sqrt(n-2) / sqrt(1 - r^2)
    """
    t_stat = rolling_ic * np.sqrt(window - 2) / np.sqrt(1 - rolling_ic**2)
    return t_stat

def evaluate_rolling_ic_tstat(data, features, window=252, plot=False):
    """
    Evaluates the rolling Information Coefficient (IC) and t-statistic for the given features.

    Parameters:
    - data: DataFrame containing the features and target variable.
    - features: List of feature names to evaluate.
    - window: Rolling window size for IC calculation (default is 252 trading days).
    - plot: Boolean indicating whether to plot the results (default is False).
    """
    IC_stats = []
    target = data['ret_1'].shift(-1).rank()

    # Momentum regime
    ma50 = data['ret_1'].rolling(50).mean()
    ma200 = data['ret_1'].rolling(200).mean()
    threshold = ma50 - ma200
    low_momentum = threshold <= 0
    high_momentum = threshold > 0

    # Volatility regime
    vol = data['ret_1'].rolling(63).std()  # 3-month realized vol
    threshold = vol.quantile(0.5)
    low_vol = vol <= threshold
    high_vol = vol > threshold
    
    for feature in features:
        # Rank the feature
        feature_rank = data[feature].rank()

        # Overall IC (non-rolling, over entire dataset)
        overall_ic = feature_rank.corr(target)
        
        # Calculate rolling IC
        rolling_ic = feature_rank.rolling(window).corr(target)
        rolling_tstat = calculate_rolling_tstat(rolling_ic, window)

        # Sign consistency
        sign = np.sign(rolling_ic).dropna()

        if sign.empty:
            sign_ratio = 0.0
            flip_rate = 0.0
            mean_run_length = 0.0
        else:
            # sign ratio: fraction of windows matching overall direction
            if overall_ic > 0:
                sign_ratio = (sign > 0).mean()
            else:
                sign_ratio = (sign < 0).mean()

            # flip rate: fraction of transitions where sign changes
            flip_rate = sign.diff().ne(0).sum() / max(1, (sign.shape[0] - 1))

            # mean run length: average length of consecutive same-sign runs
            groups = (sign != sign.shift()).cumsum()
            run_lengths = sign.groupby(groups).size()
            mean_run_length = run_lengths.mean()

        # Momentum regime ICs
        ic_low_momentum = feature_rank[low_momentum].corr(target[low_momentum])
        ic_high_momentum = feature_rank[high_momentum].corr(target[high_momentum])

        # Volatility regime ICs
        ic_low_vol = feature_rank[low_vol].corr(target[low_vol])
        ic_high_vol = feature_rank[high_vol].corr(target[high_vol])

        # Plotting
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # IC plot (top)
            rolling_ic.plot(ax=ax1)
            ax1.set_title(f'Rolling IC for {feature} (window={window})')
            ax1.set_ylabel('IC (Spearman Correlation)')
            ax1.set_xlabel('Date')
            ax1.grid(True)

            # T-stat plot (bottom)
            rolling_tstat.plot(ax=ax2, color='orange')
            ax2.set_title(f'Rolling t-stat for {feature} (window={window})')
            ax2.set_ylabel('t-statistic')
            ax2.set_xlabel('Date')
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(f'data/rollingIC_tStat/rolling_ic_tstat_{feature}.png')
            plt.close()

        # Summary stats
        IC_stats.append({
            "feature": feature,
            "overall_ic": overall_ic,
            "sign_ratio": sign_ratio,
            "flip_rate": flip_rate,
            "mean_run_length": mean_run_length,
            "mean_ic": rolling_ic.mean(),
            "std_ic": rolling_ic.std(),
            "min_ic": rolling_ic.min(),
            "max_ic": rolling_ic.max(),
            "icir": rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() != 0 else None,
            "mean_tstat": rolling_tstat.mean(),
            "std_tstat": rolling_tstat.std(),
            "min_tstat": rolling_tstat.min(),
            "max_tstat": rolling_tstat.max(),
            "ic_low_momentum": ic_low_momentum,
            "ic_high_momentum": ic_high_momentum,
            "ic_low_vol": ic_low_vol,
            "ic_high_vol": ic_high_vol
        })
    
    # Evaluate feature importance based on statistics
    IC_stats_df = pd.DataFrame(IC_stats)
    IC_stats_df['mean_ic_normalized'] = IC_stats_df['mean_ic'].abs() / IC_stats_df['mean_ic'].abs().max()
    IC_stats_df['icir_normalized'] = np.sign(IC_stats_df['mean_ic']) * IC_stats_df['icir'] / IC_stats_df['icir'].abs().max()
    IC_stats_df['sign_ratio_normalized'] = (IC_stats_df['sign_ratio'] - IC_stats_df['sign_ratio'].min()) / (IC_stats_df['sign_ratio'].max() - IC_stats_df['sign_ratio'].min()) if IC_stats_df['sign_ratio'].max() != IC_stats_df['sign_ratio'].min() else 0
    IC_stats_df['mean_run_length_normalized'] = (IC_stats_df['mean_run_length'] - IC_stats_df['mean_run_length'].min()) / (IC_stats_df['mean_run_length'].max() - IC_stats_df['mean_run_length'].min()) if IC_stats_df['mean_run_length'].max() != IC_stats_df['mean_run_length'].min() else 0
    IC_stats_df['flip_rate_normalized'] = (IC_stats_df['flip_rate'] - IC_stats_df['flip_rate'].min()) / (IC_stats_df['flip_rate'].max() - IC_stats_df['flip_rate'].min()) if IC_stats_df['flip_rate'].max() != IC_stats_df['flip_rate'].min() else 0
    IC_stats_df['score_regime'] = 1/(1+abs(IC_stats_df['ic_low_momentum'] - IC_stats_df['ic_high_momentum']) + abs(IC_stats_df['ic_low_vol'] - IC_stats_df['ic_high_vol']))
    IC_stats_df['score_regime_normalized'] = (IC_stats_df['score_regime'] - IC_stats_df['score_regime'].min()) / (IC_stats_df['score_regime'].max() - IC_stats_df['score_regime'].min()) if IC_stats_df['score_regime'].max() != IC_stats_df['score_regime'].min() else 0
    IC_stats_df['penalty_regime'] = abs(IC_stats_df['ic_low_momentum'] - IC_stats_df['ic_high_momentum']) + abs(IC_stats_df['ic_low_vol'] - IC_stats_df['ic_high_vol'])
    IC_stats_df['penalty_regime_normalized'] = (IC_stats_df['penalty_regime'] - IC_stats_df['penalty_regime'].min()) / (IC_stats_df['penalty_regime'].max() - IC_stats_df['penalty_regime'].min()) if IC_stats_df['penalty_regime'].max() != IC_stats_df['penalty_regime'].min() else 0
    IC_stats_df['penalty'] = IC_stats_df['flip_rate_normalized'] + IC_stats_df['penalty_regime_normalized']

    IC_stats_df['importance_score'] = (
        abs(IC_stats_df['mean_ic_normalized']) * 0.35 +
        abs(IC_stats_df['icir_normalized']) * 0.3 +
        IC_stats_df['sign_ratio_normalized'] * 0.15 +
        IC_stats_df['mean_run_length_normalized'] * 0.1 + IC_stats_df['score_regime_normalized'] * 0.1 - IC_stats_df['penalty'] * 0.1
    )

    # Save summary stats to CSV
    IC_stats_df.to_csv("data/feature_ic_stats.csv", index=False, float_format="%.4f")