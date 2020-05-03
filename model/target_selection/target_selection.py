# Project: Target Selection
# Description: Calculate and visualize cross-correlations between various metrics
# to determine which lead and which lag across seasons. Will help determine target
# variable for player projection modeling as we want to select the metric that leads
# all others.
# Data Sources: Basketball-Reference and ESPN
# Last Updated: 6/24/2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotting Style
plt.style.use('fivethirtyeight')

def cross_correlation(field1, field2):
    """
    Calculates the cross-correlation between two metrics to determine at which
    lag the two metrics are most correlated. For example, you could pass the function
    two identical fields (a and b) with the second lagging behind the first by one
    period.
            a = np.array([1, 2, 3, 4, 5])
            b = np.array([0, 1, 2, 3, 4])

    The function would return an integer value of 1 to represent the
    value of the lag between the two arrays.

    Args:
    field1 (array): the first metric (ex: BPM, VORP, etc.) to compare
    field2 (array): the second metric (ex: BPM, VORP, etc.) to compare

    Returns:
        Integer value representing the lag between the two fields
    """
    array_len = len(field1)
    # Take the index of the largest value in the array of correlation values calculated via a full convolve
    # cross correlation.
    arg_max = np.argmax((np.correlate([float(i) for i in field1], [float(i) for i in field2], mode='full')))
    # Map the index of the largest correlation value to that of the season lag between metrics
    return -(int(np.arange(-array_len+1, array_len)[arg_max]))

def norm_cross_correlation(field1, field2):
    """
    Calculates the cross-correlation between two metrics to determine how correlated
    the two metrics are at each period of the convolve. For example, you could pass
    the function two identical fields (a and b) with the second lagging behind
    the first by one period.
            a = np.array([1, 2, 3, 4, 5])
            b = np.array([0, 1, 2, 3, 4])

    The function would return the full discrete cross-correlation array of the
    two metrics normalized by the sum of the array. This represents the correlation
    of the two metrics at each period of the convolve.

    Args:
    field1 (array): the first metric (ex: BPM, VORP, etc.) to compare
    field2 (array): the second metric (ex: BPM, VORP, etc.) to compare

    Returns:
        List of values representing the correlation of the two metrics at each
        period of the convolve. The index with the highest value represents the
        season lag with the highest correlation.
    """
    if len(field1) > 4:
        # Select the inner nine indices of the cross-correlation array for plotting purposes
        central_corr = np.abs(np.array(np.correlate(field1, field2, mode='full'), dtype=np.float64)[len(field1)-5:len(field1)+4])
    else:
        # Select the cross-correlation array
        central_corr = np.abs(np.array(np.correlate(field1, field2, mode='full'), dtype=np.float64))
    # Normalize the cross-correlation array by the sum of the array itself
    norm_corr = np.nan_to_num(central_corr / np.sum(central_corr))
    return norm_corr.tolist()

def pad_corr_series(corr_list):
    """
    Assures each array of cross-correlation values is of the same length (9),
    padding smaller arrays with zero for plotting purposes.

    Args:
        corr_list(list): List of normalized cross-correlation values outputted from
        the norm_cross_correlation function.

    Returns:
        Array of cross-correlation values padded with zeros to meet length requirments
        for plotting.
    """
    if len(corr_list) == 9:
      return np.array(corr_list)
    elif len(corr_list) < 9:
      if len(corr_list) % 2 == 1:
        return np.pad(np.array(corr_list), pad_width=((9-len(corr_list)) // 2), mode='constant', constant_values=0.0)
      else:
        right_pad_width = int(np.floor((9-len(corr_list)) / 2))
        left_pad_width = int(np.floor((9-len(corr_list)) / 2)) + 1
        return np.pad(np.array(corr_list), pad_width=(right_pad_width, left_pad_width), mode='constant', constant_values=0.0)
    else:
      if len(corr_list) % 2 == 1:
        too_long_start = (len(corr_list) - 9) // 2
      else:
        too_long_start = (len(corr_list) - 10) // 2
      return np.array(corr_list[too_long_start:too_long_start+9])

def plot_normalized_metric(metric, df_norm):
    """
    Plots eight seperate figures each displaying the season lags between a given
    metric and all other metrics.

    Args:
        metric (string): One of the nine metrics in the metric list of 'NET_RTG',
        'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', and 'SALARY_PROP_CAP'
        df_norm (DataFrame): Pandas DataFrame containing normalized cross
        correlation.

    Returns:
        None
    """

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 10), sharex=True, sharey=True)
    metric_list = ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']
    metric_list.remove(metric)
    density_hists = [np.sum(np.vstack(tuple(pad_corr_series(x) for x in df_norm[str(metric + '_' + metric2)])), axis=0) for metric2 in metric_list]
    density_bins = np.arange(-4, 5)
    axs[0, 0].bar(density_bins, density_hists[0])
    axs[0, 1].bar(density_bins, density_hists[1])
    axs[0, 2].bar(density_bins, density_hists[2])
    axs[1, 0].bar(density_bins, density_hists[3])
    axs[1, 1].bar(density_bins, density_hists[4])
    axs[1, 2].bar(density_bins, density_hists[5])
    axs[2, 0].bar(density_bins, density_hists[6])
    axs[2, 1].bar(density_bins, density_hists[7])
    axs[1, 0].set_ylabel('Density', fontsize=14)
    axs[2, 1].set_xlabel('Season Lag', fontsize=14)
    axs[0, 0].set_title('{0} vs. {1}'.format(metric, metric_list[0]), fontsize=12)
    axs[0, 1].set_title('{0} vs. {1}'.format(metric, metric_list[1]), fontsize=12)
    axs[0, 2].set_title('{0} vs. {1}'.format(metric, metric_list[2]), fontsize=12)
    axs[1, 0].set_title('{0} vs. {1}'.format(metric, metric_list[3]), fontsize=12)
    axs[1, 1].set_title('{0} vs. {1}'.format(metric, metric_list[4]), fontsize=12)
    axs[1, 2].set_title('{0} vs. {1}'.format(metric, metric_list[5]), fontsize=12)
    axs[2, 0].set_title('{0} vs. {1}'.format(metric, metric_list[6]), fontsize=12)
    axs[2, 1].set_title('{0} vs. {1}'.format(metric, metric_list[7]), fontsize=12)
    axs[2, 2].grid(False)
    plt.suptitle('{0} Normalized Cross Correlation'.format(metric), fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

if __name__=='__main__':
    # Read in data sources
    player_table = pd.read_csv('../../../../data/player_ids/player_table.csv')
    espn_nba_rpm = pd.read_csv('../../../../data/nba/espn/espn_nba_rpm.csv')
    salary_df = pd.read_csv('../../../../data/nba/basketball_reference/player_data/salary/salary_info.csv')
    bbref_player_df = pd.read_csv('../../../../data/nba/basketball_reference/player_data/combined/bbref_player_data.csv')

    # Convert season from yyyy to yyyy-yyyy to join on
    salary_df = salary_df[salary_df['season'].notnull()]
    salary_df['season'] = salary_df.apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
    espn_nba_rpm['season'] = espn_nba_rpm.apply(lambda row: str(row['season'] - 1) +  '-' +  str(row['season']), axis=1)

    # Aggregatre ESPN metrics to season level to avoid problem joining traded players
    espn_nba_rpm = espn_nba_rpm.groupby(['name', 'pos', 'espn_link', 'season']).mean().reset_index()

    # Join dataframes
    player_data = (pd.merge(bbref_player_df, player_table, how='left', left_on='BBREF_ID', right_on='bbref_id')
                            .merge(salary_df, how='left', left_on=['bbref_id', 'SEASON'], right_on=['bbref_id', 'season'])
                            .merge(espn_nba_rpm, how='left', left_on=['espn_link', 'SEASON'], right_on=['espn_link', 'season'])
                            [['BBREF_ID', 'espn_link', 'PLAYER', 'AGE', 'MP', 'SEASON', 'TEAM', 'POSITION',
                            'PER100_ORtg', 'PER100_DRtg', 'OBPM', 'DBPM', 'BPM',
                            'VORP', 'orpm', 'drpm', 'rpm', 'wins', 'salary', 'salary_prop_cap']]
                            .rename(columns={'orpm':'ORPM', 'drpm':'DRPM', 'rpm':'RPM',
                                             'wins':'WINS', 'salary':'SALARY',
                                             'salary_prop_cap':'SALARY_PROP_CAP'}))

    # Create Net Rating metric
    player_data['NET_RTG'] = player_data['PER100_ORtg'] - player_data['PER100_DRtg']

    # Create WOR metric
    player_data['WOR'] = player_data['VORP'] * 2.7

    # Remove partial seasons resulting from trades.
    player_data_no_trades = player_data[((player_data.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')>1) &
                            (player_data['TEAM']=='TOT')) |
                            (player_data.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')<=1)]

    # Non-Normalized Cross-Correlation
    df_non_norm = player_data_no_trades.groupby(['PLAYER'])['NET_RTG'].apply(list).reset_index()
    # For each player (row) collect each metric into a list within a column
    for metric in ['RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        df_non_norm[metric] = player_data_no_trades.groupby(['PLAYER'])[metric].apply(list).reset_index()[metric]

    for metric1 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        for metric2 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
            if metric1 != metric2:
                # Create a new column for each player (row) that includes the season lag between two metrics
                df_non_norm[str(metric1 + '_' + metric2)] = df_non_norm.apply(lambda row: cross_correlation(row[metric1], row[metric2]), axis=1)

    # Aggregate to one dataframe
    corr_df = pd.DataFrame(np.arange(-15, 15, 1), columns=['SEASON_LAG'])
    for metric1 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        for metric2 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
            if metric1 != metric2:
                df = df_non_norm[str(metric1 + '_' + metric2)].value_counts().reset_index().sort_values(by='index', ascending=False)
                corr_df = pd.merge(corr_df, df, how='left', left_on='SEASON_LAG', right_on='index')
    corr_df = corr_df[[col for col in corr_df.columns if 'index' not in col]]

    # Example: Plot Histogram of Non-Normalized Lags (MP vs. BPM, VORP, NET RATING)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=True, sharey=True)
    axs[0].bar(corr_df['SEASON_LAG'], corr_df['MP_BPM'])
    axs[1].bar(corr_df['SEASON_LAG'], corr_df['MP_VORP'])
    axs[2].bar(corr_df['SEASON_LAG'], corr_df['MP_NET_RTG'])
    axs[0].set_title('Minutes Player vs. BPM', fontsize=14)
    axs[0].set_ylabel('Player Count', fontsize=14)
    axs[1].set_title('Minutes Played vs. VORP', fontsize=14)
    axs[1].set_xlabel('Season Lag', fontsize=14)
    axs[2].set_title('Minutes Played vs. Net Rating', fontsize=14)
    plt.suptitle('Cross Correlation', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Normalized Cross-Correlation
    df_norm = player_data_no_trades.groupby(['PLAYER'])['NET_RTG'].apply(list).reset_index()
    for metric in ['RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        # For each player (row) collect each metric into a list within a column
        df_norm[metric] = player_data_no_trades.groupby(['PLAYER'])[metric].apply(list).reset_index()[metric]

    for metric1 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        for metric2 in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
            if metric1 != metric2:
                # Create a new column for each player (row) that includes the cross-correlation between two metrics
                df_norm[str(metric1 + '_' + metric2)] = df_norm.apply(lambda row: norm_cross_correlation(row[metric1], row[metric2]), axis=1)


    # Example: Plot Histogram of Normalized Lags (MP vs. BPM, VORP, NET RATING)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=True, sharey=True)
    density_hist1 = np.sum(np.vstack(tuple(pad_corr_series(x) for x in df_norm['MP_BPM'])), axis=0)
    density_hist2 = np.sum(np.vstack(tuple(pad_corr_series(x) for x in df_norm['MP_VORP'])), axis=0)
    density_hist3 = np.sum(np.vstack(tuple(pad_corr_series(x) for x in df_norm['MP_NET_RTG'])), axis=0)
    density_bins = np.arange(-4, 5)
    axs[0].bar(density_bins, density_hist1)
    axs[1].bar(density_bins, density_hist2)
    axs[2].bar(density_bins, density_hist3)
    axs[0].set_title('Minutes Player vs. BPM', fontsize=14)
    axs[0].set_ylabel('Density', fontsize=14)
    axs[1].set_title('Minutes Played vs. VORP', fontsize=14)
    axs[1].set_xlabel('Season Lag', fontsize=14)
    axs[2].set_title('Minutes Played vs. NET RATING', fontsize=14)
    plt.suptitle('Normalized Cross Correlation', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Plot all cross-correlations for each individual metric
    for metric in ['NET_RTG', 'RPM', 'BPM', 'VORP', 'WOR', 'MP', 'WINS', 'SALARY', 'SALARY_PROP_CAP']:
        plot_normalized_metric(metric, df_norm)
