import matplotlib.pyplot as plt
import pandas as pd

def plot_rolling_ic(data, features, window=252):
    """
    Plots the rolling Information Coefficient (IC) for the given features.

    Parameters:
    - data: DataFrame containing the features and target variable.
    - features: List of feature names to plot.
    - window: Rolling window size for IC calculation (default is 252 trading days).
    """
    IC_stats = []
    
    for feature in features:
        # Calculate rolling IC
        rolling_ic = data[feature].rolling(window).corr(data['ret_1'].shift(-1))
        
        # Plotting
        plt.figure(figsize=(12, 4))
        rolling_ic.plot()
        plt.title(f'Rolling IC for {feature} (window={window})')
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
    
    # Save summary stats to CSV
    IC_stats_df = pd.DataFrame(IC_stats)
    IC_stats_df.to_csv("data/feature_ic_stats.csv", index=False)