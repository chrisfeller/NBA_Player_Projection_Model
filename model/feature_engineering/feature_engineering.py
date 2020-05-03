# Project: Feature Engineering
# Description: Engineer potentential features for player projection model. Join
# all features onto target variable. Impute null values.
# Data Sources: Basketball-Reference and ESPN
# Last Updated: 7/31/2019

import numpy as np
import pandas as pd

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings(action='ignore', category=SettingWithCopyWarning)

def unweighted_average(df, col):
    """
    Calculate average of previous three seasons for a given statistic. If a player
    has played fewer than three seasons then the calculation returns either the
    two-season average or current season statistic.

    Args:
        df: pandas DataFrame with statistics at the player/season level with
        partial seasons resulting from trades removed.
        col: column on with which to calculate the three-season average.

    Returns:
        df: Original pandas Dataframe with three-season average added as new
        column with the naming convention 'column_3AVG'
    """
    df['3_season_avg'] = df.groupby('BBREF_ID')[col].apply(lambda x: x.rolling(window=3).mean().round(3))
    df['2_season_avg'] = df.groupby('BBREF_ID')[col].apply(lambda x: x.rolling(window=2).mean().round(3))
    df['{}_3AVG'.format(col)] = df['3_season_avg'].fillna(df['2_season_avg']).fillna(df[col])
    df.drop(['3_season_avg', '2_season_avg'], axis=1, inplace=True)
    return df

def weight_2seasons(w):
    """
    Helper function to calculate weighted average for previous two seasons of a statistic.
    """
    def g(x):
        return (w*x).sum()/3
    return g

def weight_3season(w):
    """
    Helper function to calculate weighted average for previous three seasons of a statistic.
    """
    def g(x):
        return (w*x).sum()/5
    return g

def weighted_average(df, col):
    """
    Calculate weighted average of previous three seasons for a given statistic.
    If a player has played fewer than three seasons then the calculation returns
    either the two-season weighted average or current season statistic.

    Args:
        df: pandas DataFrame with statistics at the player/season level with
        partial seasons resulting from trades removed
        col: column on with which to calculate the three-season weighted average.

    Returns:
        df: Original pandas Dataframe with three-season weighted average added
        as new column with the naming convention 'column_3WAVG'
    """
    wts3 = np.array([1, 2, 3])
    wts2 = np.array([1, 2])
    df['3_season_avg'] = df.groupby('BBREF_ID')[col].apply(lambda x: x.rolling(window=3).apply(weight_3season(wts3), raw=True).round(3))
    df['2_season_avg'] = df.groupby('BBREF_ID')[col].apply(lambda x: x.rolling(window=2).apply(weight_2seasons(wts2), raw=True).round(3))
    df['{}_3WAVG'.format(col)] = df['3_season_avg'].fillna(df['2_season_avg']).fillna(df[col])
    df.drop(['3_season_avg', '2_season_avg'], axis=1, inplace=True)
    return df

def create_model_input(data_source_list):
    """
    Reads in a list of data sources and merges those features with the target
    variable (RPM/BPM blend) for player projection modeling, while also filling
    any null values.

    Args:
        data_source_list (list): List of one to six data sources to join onto
        the target variable. Data source options:
            - bbref_box_score: Raw, Per 100 Possession, and Advanced
                               box-score Statistics from Basketball-Reference
            - bbref_measurements: Player height and weight from Basketball-Reference
            - bbref_league_percentile: Basketball-Reference box-score statistics
                                       transformed into percentiles based on the
                                       entire league
            - bbref_position_percentile: Basketball-Reference box-score statistics
                                        transformed into percentiles based on
                                        advanced position cluster (guard, wing,
                                        big)
            - bbref_position_estimates: Positional estimate data from
                                        Basketball-Reference
            - bbref_salary: Salary data from Basketball-Reference
            - espn_advance: Advanced metrics from ESPN.com

    Returns:
        targets (pandas DataFrame): DataFrame with one or more data sources
        joined onto the target variable for player projection modeling will
        all nulls imputed.
    """
    # Dictionary of data source paths
    data_source_dict = {'targets': '../../../../data/nba/modeling_targets/modeling_targets.csv',
                        'bbref_box_score': '../../../../data/nba/basketball_reference/player_data/combined/bbref_player_data.csv',
                        'bbref_measurements': '../../../../data/nba/basketball_reference/player_data/measurements/player_measurements.csv',
                        'bbref_league_percentile': '../../../../data/nba/basketball_reference/player_data/percentile/nba_percentile_all.csv',
                        'bbref_position_percentile': '../../../../data/nba/basketball_reference/player_data/percentile/nba_percentile_position.csv',
                        'bbref_position_estimates': '../../../../data/nba/basketball_reference/player_data/positional_estimates/player_position_estimates.csv',
                        'bbref_salary': '../../../../data/nba/basketball_reference/player_data/salary/salary_info.csv',
                        'espn_advance': '../../../../data/nba/espn/espn_nba_rpm.csv'}

    # Read in Targets and reformat season to YYYY-YYYY
    targets = pd.read_csv(data_source_dict['targets'])
    targets['season'] = targets[targets['season'].notnull()].apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)

    # Join Basketball-Reference Box-Score Data to Targets if included in
    # the function parameter `data_source_list`
    if 'bbref_box_score' in data_source_list:
        # Read in Basketball-Reference Box-Score Data
        bbref_box_score = pd.read_csv(data_source_dict['bbref_box_score'])
        # Remove partial seasons resulting from in-season trades (TOT only)
        bbref_box_score = bbref_box_score[((bbref_box_score.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')>1) &
                                (bbref_box_score['TEAM']=='TOT')) |
                                (bbref_box_score.groupby(['BBREF_ID', 'SEASON'])['TEAM'].transform('size')<=1)]
        # Join onto Targets
        targets = pd.merge(targets, bbref_box_score, how='left',
                                                    left_on=['bbref_id', 'season'],
                                                    right_on=['BBREF_ID', 'SEASON'],
                                                    suffixes=('', '_duplicate'))

    # Join League Percentiles to Targets if included in the function parameter
    # `data_source_list`
    if 'bbref_league_percentile' in data_source_list:
        # Read in League Percentile Data
        bbref_league_percentile = pd.read_csv(data_source_dict['bbref_league_percentile'])
        # Join onto Targets
        targets = pd.merge(targets, bbref_league_percentile, how='left',
                                                            left_on=['bbref_id', 'season'],
                                                            right_on=['BBREF_ID', 'SEASON'],
                                                            suffixes=('', '_duplicate'))

    # Join Position Percentiles to Targets if included in the function parameter
    # `data_source_list`
    if 'bbref_position_percentile' in data_source_list:
        # Read in Position Percentile Data
        bbref_position_percentile = pd.read_csv(data_source_dict['bbref_position_percentile'])
        # Join onto Targets
        targets = pd.merge(targets, bbref_position_percentile, how='left',
                                                            left_on=['bbref_id', 'season'],
                                                            right_on=['BBREF_ID', 'SEASON'],
                                                            suffixes=('', '_duplicate'))

    # Join Measurement Data to Targets if included in the function parameter
    # `data_source_list`
    if 'bbref_measurements' in data_source_list:
        # Read in Measurable Data
        bbref_measurements = pd.read_csv(data_source_dict['bbref_measurements'])
        # Join onto Targets
        targets = pd.merge(targets, bbref_measurements, how='left',
                                                        on='bbref_id',
                                                        suffixes=('', '_duplicate'))

    # Join Salary Data to Targets if included in the function parameter
    # `data_source_list`
    if 'bbref_salary' in data_source_list:
        # Read in Salary Data and Reformat Season to YYYY-YYYY
        bbref_salary = pd.read_csv(data_source_dict['bbref_salary'])
        bbref_salary['season'] = bbref_salary[bbref_salary['season'].notnull()].apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
        # Join onto Targets
        targets = pd.merge(targets, bbref_salary, how='left',
                                                on=['bbref_id', 'season'],
                                                suffixes=('', '_duplicate'))

    # Join Positional Estimates to Targets if included in the function parameter
    # `data_source_list`
    if 'bbref_position_estimates' in data_source_list:
        # Read in Position Data and Reformat Season to YYYY-YYYY
        bbref_position_estimates = pd.read_csv(data_source_dict['bbref_position_estimates'])
        bbref_position_estimates['season'] = bbref_position_estimates.apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
        # Join onto Targets
        targets = pd.merge(targets, bbref_position_estimates, how='left',
                                                on=['bbref_id', 'season'],
                                                suffixes=('', '_duplicate'))

    # Join Advanced ESPN data to Targets if included in the function parameter
    # `data_source_list`
    if 'espn_advance' in data_source_list:
        # Read in ESPN Advance Data
        espn_advance = pd.read_csv(data_source_dict['espn_advance'])
        # Remove partial seasons resulting from in-season trades (TOT only)
        espn_advance = (espn_advance.groupby(['name', 'pos', 'espn_link', 'season'])
                                    .mean()
                                    .reset_index())
        # Join bbref_id onto espn table to join onto other dataframes
        player_table = pd.read_csv('../../../../data/player_ids/player_table.csv')
        espn_advance['season'] = espn_advance.apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
        espn_advance = (pd.merge(espn_advance, player_table,
                                    how='left', on='espn_link')
                                    [['orpm', 'drpm', 'rpm', 'wins',
                                    'bbref_id', 'season']])
        # Join onto Targets
        targets = pd.merge(targets, espn_advance, how='left',
                                                on=['bbref_id', 'season'],
                                                suffixes=('', '_duplicate'))

    # Drop duplicate fields
    targets.drop([col for col in targets.columns if '_duplicate' in col],
                    axis=1,
                        inplace=True)
    # Drop irrelivent and duplicate fields
    targets.drop([col for col in ['team_flag', 'contract_type', 'league', 'BBREF_ID', 'SEASON', 'RANK', 'POSITION_MINUTES'] if col in targets.columns], axis=1, inplace=True)
    # Impute missing values
    targets = impute_missing_values(targets)
    # Change all field names to uppercase
    targets.columns = targets.columns.str.upper()
    return targets

def impute_missing_values(df):
    """
    Imputes missing values in the model_input dataframe. Fills nulls in any shooting
    metrics with zero and nulls in non-shooting metrics with the mean of a player's
    season/advance position cluster grouping.

    Args:
        df (pandas DataFrame): DataFrame containing null values.

    Returns:
        df (pandas DataFrame): DataFrame with null values imputed.
    """
    # Define list of shooting fields in which null values will be imputed with
    # zero as they are typically a result of zero field goal attempts.
    shooting_fields = ['FG', 'FGA', 'FG%' ,'2P', '2PA', '2P%',
                        '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
                        'PER100_FG', 'PER100_FGA',
                        'PER100_FG%', 'PER100_2P', 'PER100_2PA',
                        'PER100_2P%', 'PER100_3P', 'PER100_3PA',
                        'PER100_3P%', 'PER100_FT', 'PER100_FTA',
                        'PER100_FT%', 'eFG%', 'TS%', '3PA_RATE',
                        'FT_RATE', 'fg_percentile_all', 'fga_percentile_all',
                        'fg_percent_percentile_all', 'three_point_made_percentile_all',
                        'three_point_attempt_percentile_all',
                        'three_point_percent_percentile_all',
                        'two_point_made_percentile_all',
                        'two_point_attempt_percentile_all',
                        'two_point_percent_percentile_all',
                        'efg_percent_percentile_all',
                        'true_shooting_percent_percentile_all',
                        'free_throw_made_percentile_all',
                        'free_throw_attempt_percentile_all',
                        'free_throw_percent_percentile_all',
                        'fg_made_per100_percentile_all',
                        'fg_attempted_per100_percentile_all',
                        'three_point_made_per100_percentile_all',
                        'three_point_attempt_per100_percentile_all',
                        'two_point_made_per100_percentile_all',
                        'two_point_attempt_per100_percentile_all',
                        'free_throw_made_per100_percentile_all',
                        'free_throw_attempt_per100_percentile_all',
                        'three_point_attempt_rate_percentile_all',
                        'free_throw_rate_percentile_all',
                        'fg_percentile_position',
                        'fga_percentile_position',
                        'fg_percent_percentile_position',
                        'three_point_made_percentile_position',
                        'three_point_attempt_percentile_position',
                        'three_point_percent_percentile_position',
                        'two_point_made_percentile_position',
                        'two_point_attempt_percentile_position',
                        'two_point_percent_percentile_position',
                        'efg_percent_percentile_position',
                        'true_shooting_percent_percentile_position',
                        'free_throw_made_percentile_position',
                        'free_throw_attempt_percentile_position',
                        'free_throw_percent_percentile_position',
                        'fg_made_per100_percentile_position',
                        'fg_attempted_per100_percentile_position',
                        'three_point_made_per100_percentile_position',
                        'three_point_attempt_per100_percentile_position',
                        'two_point_made_per100_percentile_position',
                        'two_point_attempt_per100_percentile_position',
                        'free_throw_made_per100_percentile_position',
                        'free_throw_attempt_per100_percentile_position',
                        'three_point_attempt_rate_percentile_position',
                        'free_throw_rate_percentile_position']

    # Define list of non-shooting fields in which nulls will be imputed with
    # the mean of the season/advance_position_cluster grouping
    box_score_fields = ['AST',
                         'AST%',
                         'BLK',
                         'BLK%',
                         'BPM',
                         'DBPM',
                         'DRB',
                         'DRB%',
                         'DWS',
                         'G',
                         'GS',
                         'IMPACT_PLAY_RATE',
                         'MP',
                         'OBPM',
                         'ORB',
                         'ORB%',
                         'OWS',
                         'PER',
                         'PER100_AST',
                         'PER100_BLK',
                         'PER100_DRB',
                         'PER100_DRtg',
                         'PER100_ORB',
                         'PER100_ORtg',
                         'PER100_PF',
                         'PER100_PTS',
                         'PER100_STL',
                         'PER100_TOV',
                         'PER100_TRB',
                         'PF',
                         'PSA',
                         'PTS',
                         'STL',
                         'STL%',
                         'TOV',
                         'TOV%',
                         'TRB',
                         'TRB%',
                         'USG%',
                         'VORP',
                         'WS',
                         'WS/48',
                         'age',
                         'age_percentile_all',
                         'age_percentile_position',
                         'ast_percent_percentile_all',
                         'ast_percent_percentile_position',
                         'ast_percentile_all',
                         'ast_percentile_position',
                         'blk_percent_percentile_all',
                         'blk_percent_percentile_position',
                         'blk_percentile_all',
                         'blk_percentile_position',
                         'bpm_percentile_all',
                         'bpm_percentile_position',
                         'def_bpm_percentile_all',
                         'def_bpm_percentile_position',
                         'def_win_shares_percentile_all',
                         'def_win_shares_percentile_position',
                         'drb_percent_percentile_all',
                         'drb_percent_percentile_position',
                         'drb_percentile_all',
                         'drb_percentile_position',
                         'experience',
                         'foul_percentile_all',
                         'foul_percentile_position',
                         'games_player_percentile_all',
                         'games_player_percentile_position',
                         'games_started_percentile_all',
                         'games_started_percentile_position',
                         'height',
                         'height_percentile_all',
                         'height_percentile_position',
                         'minutes_c',
                         'minutes_percentile_all',
                         'minutes_percentile_position',
                         'minutes_pf',
                         'minutes_pg',
                         'minutes_sf',
                         'minutes_sg',
                         'off_bpm_percentile_all',
                         'off_bpm_percentile_position',
                         'off_court_plus_minus',
                         'off_win_shares_percentile_all',
                         'off_win_shares_percentile_position',
                         'on_court_plus_minus',
                         'orb_percent_percentile_all',
                         'orb_percent_percentile_position',
                         'oreb_percentile_all',
                         'oreb_percentile_position',
                         'points_percentile_all',
                         'points_percentile_position',
                         'position_numeric',
                         'prop_c',
                         'prop_pf',
                         'prop_pg',
                         'prop_sf',
                         'prop_sg',
                         'salary',
                         'salary_prop_cap',
                         'stl_percent_percentile_all',
                         'stl_percent_percentile_position',
                         'stl_percentile_all',
                         'stl_percentile_position',
                         'total_percentile_all',
                         'total_percentile_position',
                         'tov_percent_percentile_all',
                         'tov_percent_percentile_position',
                         'trb_percent_percentile_all',
                         'trb_percent_percentile_position',
                         'turnover_percentile_all',
                         'turnover_percentile_position',
                         'usg_percent_percentile_all',
                         'usg_percent_percentile_position',
                         'vorp_percentile_all',
                         'vorp_percentile_position',
                         'weight',
                         'weight_percentile_all',
                         'weight_percentile_position',
                         'win_shares_percentile_all',
                         'win_shares_percentile_position',
                         'win_sharres_per_48_percentile_all',
                         'win_sharres_per_48_percentile_position']

    # Check to see if `advanced_position_cluster` was joined onto Targets dataframe
    # in the create_model_input function. Will use `advanced_position_cluster` field
    # to groupby when filling null values to impute mean of season/position.
    if 'advanced_position_cluster' in df.columns:
        # Impute shooting fields with zero
        df.update(df[[col for col in df.columns if col in shooting_fields]].fillna(0))
        # Impute non-shooting fields with mean of season/advance_position_cluster
        df[[col for col in df.columns if col in box_score_fields]] = df.groupby(['advanced_position_cluster', 'season'])[[col for col in df.columns if col in box_score_fields]].transform(lambda x: x.fillna(x.mean()))
        return df

    else:
        # Read in Position Data and Reformat Season to YYYY-YYYY
        bbref_position_estimates = pd.read_csv('../../../../data/nba/basketball_reference/player_data/positional_estimates/player_position_estimates.csv')
        bbref_position_estimates['season'] = bbref_position_estimates.apply(lambda row: str(int(row['season'] - 1)) +  '-' +  str(int(row['season'])), axis=1)
        bbref_position_estimates = bbref_position_estimates[['bbref_id', 'season', 'advanced_position_cluster']]
        # Join onto Targets
        df = pd.merge(df, bbref_position_estimates, how='left',
                                                on=['bbref_id', 'season'],
                                                suffixes=('', '_duplicate'))
        df.drop([col for col in df.columns if '_duplicate' in col], axis=1, inplace=True)

        # Impute shooting fields with zero
        df.update(df[[col for col in df.columns if col in shooting_fields]].fillna(0))
        # Impute non-shooting fields with mean of season/advance_position_cluster
        df[[col for col in df.columns if col in box_score_fields]] = df.groupby(['advanced_position_cluster', 'season'])[[col for col in df.columns if col in box_score_fields]].transform(lambda x: x.fillna(x.mean()))
        return df

def metrics_to_averages(df, weighted=True):
    """
    Transforms fields from season-level statistics to either unweighted or weighted
    three-season averages.

    Args:
        df (pandas DataFrame): DataFrame containing season-level statistics
        weighted (boolean): Whether to weight the statistical average (Default=True)

    Returns:
        df (pandas DataFrame): DataFrame with season-level statistics transformed
        to either unweighted or weighted three-season averages.
    """
    # Define comprehensive list of metrics to transform from season-level to
    # either unweighted or weighted three-season averages
    metric_fields = ['G',
                    'GS',
                    'MP',
                    'FG',
                    'FGA',
                    'FG%',
                    '3P',
                    '3PA',
                    '3P%',
                    '2P',
                    '2PA',
                    '2P%',
                    'EFG%',
                    'FT',
                    'FTA',
                    'FT%',
                    'ORB',
                    'DRB',
                    'TRB',
                    'AST',
                    'STL',
                    'BLK',
                    'TOV',
                    'PF',
                    'PTS',
                    'PER100_FG',
                    'PER100_FGA',
                    'PER100_FG%',
                    'PER100_3P',
                    'PER100_3PA',
                    'PER100_3P%',
                    'PER100_2P',
                    'PER100_2PA',
                    'PER100_2P%',
                    'PER100_FT',
                    'PER100_FTA',
                    'PER100_FT%',
                    'PER100_ORB',
                    'PER100_DRB',
                    'PER100_TRB',
                    'PER100_AST',
                    'PER100_STL',
                    'PER100_BLK',
                    'PER100_TOV',
                    'PER100_PF',
                    'PER100_PTS',
                    'PER100_ORTG',
                    'PER100_DRTG',
                    'PER',
                    'TS%',
                    '3PA_RATE',
                    'FT_RATE',
                    'ORB%',
                    'DRB%',
                    'TRB%',
                    'AST%',
                    'STL%',
                    'BLK%',
                    'TOV%',
                    'USG%',
                    'OWS',
                    'DWS',
                    'WS',
                    'WS/48',
                    'OBPM',
                    'DBPM',
                    'BPM',
                    'VORP',
                    'GAMES_PLAYER_PERCENTILE_ALL',
                    'GAMES_STARTED_PERCENTILE_ALL',
                    'MINUTES_PERCENTILE_ALL',
                    'FG_PERCENTILE_ALL',
                    'FGA_PERCENTILE_ALL',
                    'FG_PERCENT_PERCENTILE_ALL',
                    'THREE_POINT_MADE_PERCENTILE_ALL',
                    'THREE_POINT_ATTEMPT_PERCENTILE_ALL',
                    'THREE_POINT_PERCENT_PERCENTILE_ALL',
                    'TWO_POINT_MADE_PERCENTILE_ALL',
                    'TWO_POINT_ATTEMPT_PERCENTILE_ALL',
                    'TWO_POINT_PERCENT_PERCENTILE_ALL',
                    'EFG_PERCENT_PERCENTILE_ALL',
                    'TRUE_SHOOTING_PERCENT_PERCENTILE_ALL',
                    'FREE_THROW_MADE_PERCENTILE_ALL',
                    'FREE_THROW_ATTEMPT_PERCENTILE_ALL',
                    'FREE_THROW_PERCENT_PERCENTILE_ALL',
                    'OREB_PERCENTILE_ALL',
                    'DRB_PERCENTILE_ALL',
                    'TOTAL_PERCENTILE_ALL',
                    'AST_PERCENTILE_ALL',
                    'STL_PERCENTILE_ALL',
                    'BLK_PERCENTILE_ALL',
                    'TURNOVER_PERCENTILE_ALL',
                    'FOUL_PERCENTILE_ALL',
                    'POINTS_PERCENTILE_ALL',
                    'FG_MADE_PER100_PERCENTILE_ALL',
                    'FG_ATTEMPTED_PER100_PERCENTILE_ALL',
                    'THREE_POINT_MADE_PER100_PERCENTILE_ALL',
                    'THREE_POINT_ATTEMPT_PER100_PERCENTILE_ALL',
                    'TWO_POINT_MADE_PER100_PERCENTILE_ALL',
                    'TWO_POINT_ATTEMPT_PER100_PERCENTILE_ALL',
                    'FREE_THROW_MADE_PER100_PERCENTILE_ALL',
                    'FREE_THROW_ATTEMPT_PER100_PERCENTILE_ALL',
                    'THREE_POINT_ATTEMPT_RATE_PERCENTILE_ALL',
                    'FREE_THROW_RATE_PERCENTILE_ALL',
                    'ORB_PERCENT_PERCENTILE_ALL',
                    'DRB_PERCENT_PERCENTILE_ALL',
                    'TRB_PERCENT_PERCENTILE_ALL',
                    'AST_PERCENT_PERCENTILE_ALL',
                    'STL_PERCENT_PERCENTILE_ALL',
                    'BLK_PERCENT_PERCENTILE_ALL',
                    'TOV_PERCENT_PERCENTILE_ALL',
                    'USG_PERCENT_PERCENTILE_ALL',
                    'OFF_WIN_SHARES_PERCENTILE_ALL',
                    'DEF_WIN_SHARES_PERCENTILE_ALL',
                    'WIN_SHARES_PERCENTILE_ALL',
                    'WIN_SHARRES_PER_48_PERCENTILE_ALL',
                    'OFF_BPM_PERCENTILE_ALL',
                    'DEF_BPM_PERCENTILE_ALL',
                    'BPM_PERCENTILE_ALL',
                    'VORP_PERCENTILE_ALL',
                    'GAMES_PLAYER_PERCENTILE_POSITION',
                    'GAMES_STARTED_PERCENTILE_POSITION',
                    'MINUTES_PERCENTILE_POSITION',
                    'FG_PERCENTILE_POSITION',
                    'FGA_PERCENTILE_POSITION',
                    'FG_PERCENT_PERCENTILE_POSITION',
                    'THREE_POINT_MADE_PERCENTILE_POSITION',
                    'THREE_POINT_ATTEMPT_PERCENTILE_POSITION',
                    'THREE_POINT_PERCENT_PERCENTILE_POSITION',
                    'TWO_POINT_MADE_PERCENTILE_POSITION',
                    'TWO_POINT_ATTEMPT_PERCENTILE_POSITION',
                    'TWO_POINT_PERCENT_PERCENTILE_POSITION',
                    'EFG_PERCENT_PERCENTILE_POSITION',
                    'TRUE_SHOOTING_PERCENT_PERCENTILE_POSITION',
                    'FREE_THROW_MADE_PERCENTILE_POSITION',
                    'FREE_THROW_ATTEMPT_PERCENTILE_POSITION',
                    'FREE_THROW_PERCENT_PERCENTILE_POSITION',
                    'OREB_PERCENTILE_POSITION',
                    'DRB_PERCENTILE_POSITION',
                    'TOTAL_PERCENTILE_POSITION',
                    'AST_PERCENTILE_POSITION',
                    'STL_PERCENTILE_POSITION',
                    'BLK_PERCENTILE_POSITION',
                    'TURNOVER_PERCENTILE_POSITION',
                    'FOUL_PERCENTILE_POSITION',
                    'POINTS_PERCENTILE_POSITION',
                    'FG_MADE_PER100_PERCENTILE_POSITION',
                    'FG_ATTEMPTED_PER100_PERCENTILE_POSITION',
                    'THREE_POINT_MADE_PER100_PERCENTILE_POSITION',
                    'THREE_POINT_ATTEMPT_PER100_PERCENTILE_POSITION',
                    'TWO_POINT_MADE_PER100_PERCENTILE_POSITION',
                    'TWO_POINT_ATTEMPT_PER100_PERCENTILE_POSITION',
                    'FREE_THROW_MADE_PER100_PERCENTILE_POSITION',
                    'FREE_THROW_ATTEMPT_PER100_PERCENTILE_POSITION',
                    'THREE_POINT_ATTEMPT_RATE_PERCENTILE_POSITION',
                    'FREE_THROW_RATE_PERCENTILE_POSITION',
                    'ORB_PERCENT_PERCENTILE_POSITION',
                    'DRB_PERCENT_PERCENTILE_POSITION',
                    'TRB_PERCENT_PERCENTILE_POSITION',
                    'AST_PERCENT_PERCENTILE_POSITION',
                    'STL_PERCENT_PERCENTILE_POSITION',
                    'BLK_PERCENT_PERCENTILE_POSITION',
                    'TOV_PERCENT_PERCENTILE_POSITION',
                    'USG_PERCENT_PERCENTILE_POSITION',
                    'OFF_WIN_SHARES_PERCENTILE_POSITION',
                    'DEF_WIN_SHARES_PERCENTILE_POSITION',
                    'WIN_SHARES_PERCENTILE_POSITION',
                    'WIN_SHARRES_PER_48_PERCENTILE_POSITION',
                    'OFF_BPM_PERCENTILE_POSITION',
                    'DEF_BPM_PERCENTILE_POSITION',
                    'BPM_PERCENTILE_POSITION',
                    'VORP_PERCENTILE_POSITION',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'GAMES_PLAYED',
                    'MINUTES_PLAYED',
                    'ON_COURT_PLUS_MINUS',
                    'OFF_COURT_PLUS_MINUS',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'MINUTES_PG',
                    'MINUTES_SG',
                    'MINUTES_SF',
                    'MINUTES_PF',
                    'MINUTES_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']
    if weighted:
        # Create three-season weighted average columns
        for col in [col for col in df.columns if col in metric_fields]:
            weighted_average(df, col)
        # Drop original season-level columns
        df.drop([col for col in df.columns if col in metric_fields], axis=1, inplace=True)
    else:
        # Create three-season un-weighted average columns
        for col in [col for col in df.columns if col in metric_fields]:
            unweighted_average(df, col)
        # Drop original season-level columns
        df.drop([col for col in df.columns if col in metric_fields], axis=1, inplace=True)
    return df

if __name__=='__main__':
    # Transform single-season features into three-season weighted averages
    model_input = create_model_input(['bbref_box_score',
                                      'bbref_measurements',
                                      'bbref_league_percentile',
                                      'bbref_position_percentile',
                                      'bbref_position_estimates',
                                      'bbref_salary',
                                      'espn_advance'])
    model_input_3WAVG = metrics_to_averages(model_input)

    # Create single-season features from Box Score, League Percentiles,
    # Position_Percentiles, ESPN Advance, Positional Estimates, Measurements,
    # and Salary data sources
    model_input = create_model_input(['bbref_box_score',
                                      'bbref_measurements',
                                      'bbref_league_percentile',
                                      'bbref_position_percentile',
                                      'bbref_position_estimates',
                                      'bbref_salary',
                                      'espn_advance'])

    # Join single-season and three-season weighted average features into single
    # feature matrix to use in model_selection and model_pipeline scripts
    complete_feature_matrix = pd.merge(model_input, model_input_3WAVG,
                                        on=['BBREF_ID', 'SEASON'],
                                        suffixes=('', '_duplicate'))
    complete_feature_matrix.drop([col for col in complete_feature_matrix.columns if '_duplicate' in col],
                                axis=1,
                                inplace=True)
    complete_feature_matrix.to_csv('../feature_selection/featurized_inputs/complete_feature_matrix.csv',
                                    index=False)
