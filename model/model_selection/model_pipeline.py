# Project: Player Projection Model (Chris)
# Description: Build a pipeline to gridsearch over various data inputs,
# model types, and hyperparameters. Build five seperate models to predict
# a player's future RPM/BPM Blend +1 to +5 seasons into the future.
# Data Sources: Basketball-Reference and ESPN
# Last Updated: 8/3/2019

import numpy as np
import pandas as pd
import imgkit
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

# Supress various warnings. Warnings won't surpress when gridsearch is run in
# parallel
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

class CustomSelector(TransformerMixin, BaseEstimator):
    '''
    Custom Transformer used in pipeline to select a subset of predictors. Used
    in GridSearchCV process to determine which predictor subsets are most
    predictive for each season+ model.

    Predictor Subsets:
        - box_score: Single-Season Per-100 Possession and Advanced Metrics
        - box_score_3WAVG: Three-season weighted averages for Per-100 Possession
                           and Advanced Metrics
        - league_percentiles: Single-season percentile of a player's performance
                              in a given metric compared to the entire league
        - league_percentiles_3WAVG: Three-season weighted average percentile of
                                    a player's performance in a given metric
                                    compared to the entire league
        - position_percentiles: Single-season percentile of a player's performance
                            in a given metric compared to the player's advanced
                            cluster position (Guard, Wing, Big)
        - position_percentiles_3WAVG: Three-season weighted average percentile
                                      of a player's performance in a given
                                      metric compared to the player's advanced
                                      cluster position (Guard, Wing, Big)
    '''

    def __init__(self, data_subset='box_score'):
        self.data_subset = data_subset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.data_subset == 'box_score':
            return X[['AGE',
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
                    'HEIGHT',
                    'WEIGHT',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'EXPERIENCE',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']]
        elif self.data_subset == 'box_score_3WAVG':
            return X[['AGE',
                    'PER100_FG_3WAVG',
                    'PER100_FGA_3WAVG',
                    'PER100_FG%_3WAVG',
                    'PER100_3P_3WAVG',
                    'PER100_3PA_3WAVG',
                    'PER100_3P%_3WAVG',
                    'PER100_2P_3WAVG',
                    'PER100_2PA_3WAVG',
                    'PER100_2P%_3WAVG',
                    'PER100_FT_3WAVG',
                    'PER100_FTA_3WAVG',
                    'PER100_FT%_3WAVG',
                    'PER100_ORB_3WAVG',
                    'PER100_DRB_3WAVG',
                    'PER100_TRB_3WAVG',
                    'PER100_AST_3WAVG',
                    'PER100_STL_3WAVG',
                    'PER100_BLK_3WAVG',
                    'PER100_TOV_3WAVG',
                    'PER100_PF_3WAVG',
                    'PER100_PTS_3WAVG',
                    'PER100_ORTG_3WAVG',
                    'PER100_DRTG_3WAVG',
                    'PER_3WAVG',
                    'TS%_3WAVG',
                    '3PA_RATE_3WAVG',
                    'FT_RATE_3WAVG',
                    'ORB%_3WAVG',
                    'DRB%_3WAVG',
                    'TRB%_3WAVG',
                    'AST%_3WAVG',
                    'STL%_3WAVG',
                    'BLK%_3WAVG',
                    'TOV%_3WAVG',
                    'USG%_3WAVG',
                    'OWS_3WAVG',
                    'DWS_3WAVG',
                    'WS_3WAVG',
                    'WS/48_3WAVG',
                    'OBPM_3WAVG',
                    'DBPM_3WAVG',
                    'BPM_3WAVG',
                    'VORP_3WAVG',
                    'HEIGHT',
                    'WEIGHT',
                    'SALARY_3WAVG',
                    'SALARY_PROP_CAP_3WAVG',
                    'EXPERIENCE',
                    'PROP_PG_3WAVG',
                    'PROP_SG_3WAVG',
                    'PROP_SF_3WAVG',
                    'PROP_PF_3WAVG',
                    'PROP_C_3WAVG',
                    'POSITION_NUMERIC_3WAVG',
                    'ORPM_3WAVG',
                    'DRPM_3WAVG',
                    'RPM_3WAVG',
                    'WINS_3WAVG']]
        elif self.data_subset == 'league_percentiles':
            return X[['AGE_PERCENTILE_ALL',
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
                    'HEIGHT_PERCENTILE_ALL',
                    'WEIGHT_PERCENTILE_ALL',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'EXPERIENCE',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']]
        elif self.data_subset == 'league_percentiles_3WAVG':
            return X[['AGE_PERCENTILE_ALL',
                    'GAMES_PLAYER_PERCENTILE_ALL_3WAVG',
                    'GAMES_STARTED_PERCENTILE_ALL_3WAVG',
                    'MINUTES_PERCENTILE_ALL_3WAVG',
                    'FG_PERCENTILE_ALL_3WAVG',
                    'FGA_PERCENTILE_ALL_3WAVG',
                    'FG_PERCENT_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_MADE_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_ATTEMPT_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_PERCENT_PERCENTILE_ALL_3WAVG',
                    'TWO_POINT_MADE_PERCENTILE_ALL_3WAVG',
                    'TWO_POINT_ATTEMPT_PERCENTILE_ALL_3WAVG',
                    'TWO_POINT_PERCENT_PERCENTILE_ALL_3WAVG',
                    'EFG_PERCENT_PERCENTILE_ALL_3WAVG',
                    'TRUE_SHOOTING_PERCENT_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_MADE_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_ATTEMPT_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_PERCENT_PERCENTILE_ALL_3WAVG',
                    'OREB_PERCENTILE_ALL_3WAVG',
                    'DRB_PERCENTILE_ALL_3WAVG',
                    'TOTAL_PERCENTILE_ALL_3WAVG',
                    'AST_PERCENTILE_ALL_3WAVG',
                    'STL_PERCENTILE_ALL_3WAVG',
                    'BLK_PERCENTILE_ALL_3WAVG',
                    'TURNOVER_PERCENTILE_ALL_3WAVG',
                    'FOUL_PERCENTILE_ALL_3WAVG',
                    'POINTS_PERCENTILE_ALL_3WAVG',
                    'FG_MADE_PER100_PERCENTILE_ALL_3WAVG',
                    'FG_ATTEMPTED_PER100_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_MADE_PER100_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_ATTEMPT_PER100_PERCENTILE_ALL_3WAVG',
                    'TWO_POINT_MADE_PER100_PERCENTILE_ALL_3WAVG',
                    'TWO_POINT_ATTEMPT_PER100_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_MADE_PER100_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_ATTEMPT_PER100_PERCENTILE_ALL_3WAVG',
                    'THREE_POINT_ATTEMPT_RATE_PERCENTILE_ALL_3WAVG',
                    'FREE_THROW_RATE_PERCENTILE_ALL_3WAVG',
                    'ORB_PERCENT_PERCENTILE_ALL_3WAVG',
                    'DRB_PERCENT_PERCENTILE_ALL_3WAVG',
                    'TRB_PERCENT_PERCENTILE_ALL_3WAVG',
                    'AST_PERCENT_PERCENTILE_ALL_3WAVG',
                    'STL_PERCENT_PERCENTILE_ALL_3WAVG',
                    'BLK_PERCENT_PERCENTILE_ALL_3WAVG',
                    'TOV_PERCENT_PERCENTILE_ALL_3WAVG',
                    'USG_PERCENT_PERCENTILE_ALL_3WAVG',
                    'OFF_WIN_SHARES_PERCENTILE_ALL_3WAVG',
                    'DEF_WIN_SHARES_PERCENTILE_ALL_3WAVG',
                    'WIN_SHARES_PERCENTILE_ALL_3WAVG',
                    'WIN_SHARRES_PER_48_PERCENTILE_ALL_3WAVG',
                    'OFF_BPM_PERCENTILE_ALL_3WAVG',
                    'DEF_BPM_PERCENTILE_ALL_3WAVG',
                    'BPM_PERCENTILE_ALL_3WAVG',
                    'VORP_PERCENTILE_ALL_3WAVG',
                    'HEIGHT_PERCENTILE_ALL',
                    'WEIGHT_PERCENTILE_ALL',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'EXPERIENCE',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']]
        elif self.data_subset == 'position_percentiles':
            return X[['AGE_PERCENTILE_POSITION',
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
                    'HEIGHT_PERCENTILE_POSITION',
                    'WEIGHT_PERCENTILE_POSITION',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'EXPERIENCE',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']]
        elif self.data_subset == 'position_percentiles_3WAVG':
            return X[['AGE_PERCENTILE_POSITION',
                    'GAMES_PLAYER_PERCENTILE_POSITION_3WAVG',
                    'GAMES_STARTED_PERCENTILE_POSITION_3WAVG',
                    'MINUTES_PERCENTILE_POSITION_3WAVG',
                    'FG_PERCENTILE_POSITION_3WAVG',
                    'FGA_PERCENTILE_POSITION_3WAVG',
                    'FG_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_MADE_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_ATTEMPT_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'TWO_POINT_MADE_PERCENTILE_POSITION_3WAVG',
                    'TWO_POINT_ATTEMPT_PERCENTILE_POSITION_3WAVG',
                    'TWO_POINT_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'EFG_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'TRUE_SHOOTING_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_MADE_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_ATTEMPT_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'OREB_PERCENTILE_POSITION_3WAVG',
                    'DRB_PERCENTILE_POSITION_3WAVG',
                    'TOTAL_PERCENTILE_POSITION_3WAVG',
                    'AST_PERCENTILE_POSITION_3WAVG',
                    'STL_PERCENTILE_POSITION_3WAVG',
                    'BLK_PERCENTILE_POSITION_3WAVG',
                    'TURNOVER_PERCENTILE_POSITION_3WAVG',
                    'FOUL_PERCENTILE_POSITION_3WAVG',
                    'POINTS_PERCENTILE_POSITION_3WAVG',
                    'FG_MADE_PER100_PERCENTILE_POSITION_3WAVG',
                    'FG_ATTEMPTED_PER100_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_MADE_PER100_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_ATTEMPT_PER100_PERCENTILE_POSITION_3WAVG',
                    'TWO_POINT_MADE_PER100_PERCENTILE_POSITION_3WAVG',
                    'TWO_POINT_ATTEMPT_PER100_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_MADE_PER100_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_ATTEMPT_PER100_PERCENTILE_POSITION_3WAVG',
                    'THREE_POINT_ATTEMPT_RATE_PERCENTILE_POSITION_3WAVG',
                    'FREE_THROW_RATE_PERCENTILE_POSITION_3WAVG',
                    'ORB_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'DRB_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'TRB_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'AST_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'STL_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'BLK_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'TOV_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'USG_PERCENT_PERCENTILE_POSITION_3WAVG',
                    'OFF_WIN_SHARES_PERCENTILE_POSITION_3WAVG',
                    'DEF_WIN_SHARES_PERCENTILE_POSITION_3WAVG',
                    'WIN_SHARES_PERCENTILE_POSITION_3WAVG',
                    'WIN_SHARRES_PER_48_PERCENTILE_POSITION_3WAVG',
                    'OFF_BPM_PERCENTILE_POSITION_3WAVG',
                    'DEF_BPM_PERCENTILE_POSITION_3WAVG',
                    'BPM_PERCENTILE_POSITION_3WAVG',
                    'VORP_PERCENTILE_POSITION_3WAVG',
                    'HEIGHT_PERCENTILE_POSITION',
                    'WEIGHT_PERCENTILE_POSITION',
                    'SALARY',
                    'SALARY_PROP_CAP',
                    'EXPERIENCE',
                    'PROP_PG',
                    'PROP_SG',
                    'PROP_SF',
                    'PROP_PF',
                    'PROP_C',
                    'POSITION_NUMERIC',
                    'ORPM',
                    'DRPM',
                    'RPM',
                    'WINS']]


if __name__=='__main__':
    # Read in full dataset containing single-season and three-season weighted
    # averages for box score, league percentiles, position percentiles,
    # measurements, salary, and espn advanced data
    complete_feature_matrix = pd.read_csv('../feature_selection/featurized_inputs/complete_feature_matrix.csv')

    # Instantiate Scoring and Parameter Dicitonaries to hold model outputs
    cross_val_scores = {}
    test_scores = {}
    model_best_params = {}

    # Loop through +1 to +5 seasons
    for future_season in np.arange(1, 6):
        # Filter to correct target_variable (RPM/BPM Blend)
        df = (complete_feature_matrix[complete_feature_matrix['SEASON_PLUS_{0}'.format(future_season)].notnull()])
        # Train/Test Split
        y = df.pop('SEASON_PLUS_{0}'.format(future_season))
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=10)
        # Instantiate Pipeline with three steps: 1. Predictor Subset Selector
        # 2. Standard Scaler 3. Model Selection
        full_pipeline = Pipeline([
            ('feature_selection', CustomSelector()),
            ('scaler', StandardScaler()),
            ('estimator', Ridge())])

        # Set parameter list to GridSearch over
        # To add a model parameter: `'estimator__parameter': [value1, value2]`
        # Note the double underscore in the above syntax
        # Resource: https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn
        param_list = [{'feature_selection__data_subset': ['box_score',
                                                          'box_score_3WAVG',
                                                          'league_percentiles',
                                                          'league_percentiles_3WAVG',
                                                          'position_percentiles',
                                                          'position_percentiles_3WAVG'],
                        'estimator': [Ridge()]},
                        {'feature_selection__data_subset': ['box_score',
                                                          'box_score_3WAVG',
                                                          'league_percentiles',
                                                          'league_percentiles_3WAVG',
                                                          'position_percentiles',
                                                          'position_percentiles_3WAVG'],
                        'estimator': [Lasso()]},
                        {'feature_selection__data_subset': ['box_score',
                                                          'box_score_3WAVG',
                                                          'league_percentiles',
                                                          'league_percentiles_3WAVG',
                                                          'position_percentiles',
                                                          'position_percentiles_3WAVG'],
                        'estimator': [ElasticNet()]},
                        {'feature_selection__data_subset': ['box_score',
                                                          'box_score_3WAVG',
                                                          'league_percentiles',
                                                          'league_percentiles_3WAVG',
                                                          'position_percentiles',
                                                          'position_percentiles_3WAVG'],
                        'estimator': [RandomForestRegressor()]},
                        {'feature_selection__data_subset': ['box_score',
                                                          'box_score_3WAVG',
                                                          'league_percentiles',
                                                          'league_percentiles_3WAVG',
                                                          'position_percentiles',
                                                          'position_percentiles_3WAVG'],
                        'estimator': [GradientBoostingRegressor()]}]

        # Instantiate GridSearchCV object
        grid = GridSearchCV(full_pipeline, param_grid=param_list,
                                           scoring='neg_mean_squared_error',
                                           cv=5,
                                           n_jobs=-1)
        # Fit GridSearch object
        grid.fit(X_train, y_train)
        # Print best cross validation score from gridsearch process
        print('Cross Validation RMSE: ', np.sqrt(abs(grid.best_score_)))
        # Save best cross validation score in dictionary
        cross_val_scores['Season+{0}'.format(future_season)] = np.sqrt(abs(grid.best_score_))
        # Print best model parameters from gridsearch process
        print('Best Params: ', grid.best_params_)
        # Save best model parameters in dictionary
        model_best_params['Season+{0}'.format(future_season)] = grid.best_params_

        # Save pipeline
        # Can load pipeline via following syntax
        # ```
        # pipeline = joblib.load('models/season+1_pipeline.pkl')
        # y_pred = pipeline.predict(X_test)
        # ```
        joblib.dump(grid.best_estimator_, 'models/season+{0}_pipeline.pkl'.format(future_season))

        # Score Test Set
        y_pred = grid.predict(X_test)
        print('Test RMSE: ', np.sqrt(abs(mean_squared_error(y_test, y_pred))))
        # Save test set score to dictionary
        test_scores['Season+{0}'.format(future_season)] = np.sqrt(abs(mean_squared_error(y_test, y_pred)))

        # Make predictions on full dataset
        y = complete_feature_matrix['SEASON_PLUS_{0}'.format(future_season)]
        X = complete_feature_matrix.copy()
        y_pred = grid.predict(X)
        complete_feature_matrix['PLUS{0}_PREDICTION'.format(future_season)] = y_pred
    # Combine +1 through +5 predicitons with original dataframe
    predictions_df = complete_feature_matrix[['BBREF_ID',
                                               'PLAYER',
                                               'SEASON',
                                               'ADVANCED_POSITION_CLUSTER',
                                               'PLUS1_PREDICTION',
                                               'PLUS2_PREDICTION',
                                               'PLUS3_PREDICTION',
                                               'PLUS4_PREDICTION',
                                               'PLUS5_PREDICTION']]

    # Write out predictions
    predictions_df.to_csv('predictions/predictions.csv', index=False)

    # Combine actuals and future predictions
    future_predictions = predictions_df[predictions_df['SEASON']=='2018-2019']
    actuals = complete_feature_matrix[['BBREF_ID', 'SEASON', 'BLEND']]
    # Transform future predictions from wide to long format
    future_predictions_long = pd.melt(future_predictions[['BBREF_ID',
                                    'PLUS1_PREDICTION',
                                    'PLUS2_PREDICTION',
                                    'PLUS3_PREDICTION',
                                    'PLUS4_PREDICTION',
                                    'PLUS5_PREDICTION']],
                                    id_vars=['BBREF_ID'],
                                    value_vars=['PLUS1_PREDICTION',
                                                'PLUS2_PREDICTION',
                                                'PLUS3_PREDICTION',
                                                'PLUS4_PREDICTION',
                                                'PLUS5_PREDICTION'])
    # Add yyyy-yyyy values to long-form predictions
    replace_dict = {'PLUS1_PREDICTION': '2019-2020',
                    'PLUS2_PREDICTION': '2020-2021',
                    'PLUS3_PREDICTION': '2021-2022',
                    'PLUS4_PREDICTION': '2022-2023',
                    'PLUS5_PREDICTION': '2023-2024'}
    future_predictions_long['SEASON'] = future_predictions_long['variable'].replace(replace_dict)
    future_predictions_long.drop('variable', axis=1, inplace=True)
    future_predictions_long.rename(columns={'value': 'BLEND'}, inplace=True)
    # Combine actuals and future predictions
    result = pd.concat([actuals, future_predictions_long])
    result.rename(columns={'BBREF_ID': 'bbref_id', 'SEASON': 'season', 'BLEND': 'blend'},
                  inplace=True)
    # Write to predictions folder
    result.to_csv('predictions/actuals_and_predictions.csv', index=False)

    # Save top-25 predictions for 2018-2019 in pandas styling format
    styled_top25 = (predictions_df[predictions_df['SEASON']=='2018-2019']
                     .sort_values(by='PLUS1_PREDICTION',
                                  ascending=False)
                     .iloc[0:25, 1:5]
                     .style
                     .set_table_styles(
                     [{'selector': 'tr:nth-of-type(odd)',
                       'props': [('background', '#eee')]},
                      {'selector': 'tr:nth-of-type(even)',
                       'props': [('background', 'white')]},
                      {'selector':'th, td', 'props':[('text-align', 'center')]}])
                     .set_properties(subset=['PLAYER',
                                            'AGE',
                                            'SEASON',
                                            'ADVANCED_POSITION_CLUSTER'],
                                    **{'text-align': 'left'})
                     .hide_index()
                     .background_gradient(subset=['PLUS1_PREDICTION'], cmap='Reds'))
    html = styled_top25.render()
    imgkit.from_string(html, 'plots/top_25predictions.png', {'width': 1})

    # Save bottom-25 predictions for 2018-2019 in pandas styling format
    styled_bottom25 = (predictions_df[predictions_df['SEASON']=='2018-2019']
                     .sort_values(by='PLUS1_PREDICTION',
                                  ascending=False)
                     .iloc[-25:, 1:5]
                     .style
                     .set_table_styles(
                     [{'selector': 'tr:nth-of-type(odd)',
                       'props': [('background', '#eee')]},
                      {'selector': 'tr:nth-of-type(even)',
                       'props': [('background', 'white')]},
                      {'selector':'th, td', 'props':[('text-align', 'center')]}])
                     .set_properties(subset=['PLAYER',
                                            'AGE',
                                            'SEASON',
                                            'ADVANCED_POSITION_CLUSTER'],
                                    **{'text-align': 'left'})
                     .hide_index()
                     .background_gradient(subset=['PLUS1_PREDICTION'], cmap='Blues_r'))
    html = styled_bottom25.render()
    imgkit.from_string(html, 'plots/bottom_25predictions.png', {'width': 1})

    # Create Model Performance Tables
    # Cross validation scores from models +1 through +5
    cross_val_scores_df = pd.DataFrame.from_dict(cross_val_scores, orient='index').reset_index()
    cross_val_scores_df.columns = ['SEASON', 'CV_RMSE']
    # Test scores from models +1 through +5
    test_scores_df = pd.DataFrame.from_dict(test_scores, orient='index').reset_index()
    test_scores_df.columns = ['SEASON', 'TEST_RMSE']
    # Join Cross Validation and Test Scores into one dataframe
    scores_df = pd.merge(cross_val_scores_df, test_scores_df, on='SEASON', how='inner')

    # Create Model Type/Parameters Table
    best_params_df = pd.DataFrame.from_dict(model_best_params, orient='index').reset_index()[['index', 'feature_selection__data_subset']]
    best_params_df['MODEL_TYPE'] = np.array([str(model_best_params['Season+{0}'.format(i)]['estimator']).split('(')[0] for i in range(1, 6)])
    best_params_df.columns = ['SEASON', 'PREDICTOR_SUBSET', 'MODEL_TYPE']

    # Join Model Scores and Parameter Tables
    model_performance_df = pd.merge(scores_df, best_params_df, on='SEASON')
    # Style table and save to .png file for readme
    model_performance_df_styled = (model_performance_df
                                    .style
                                    .set_table_styles(
                                    [{'selector': 'tr:nth-of-type(odd)',
                                      'props': [('background', '#eee')]},
                                     {'selector': 'tr:nth-of-type(even)',
                                      'props': [('background', 'white')]},
                                     {'selector':'th, td', 'props':[('text-align', 'center')]}])
                                     .set_properties(subset=['MODEL_TYPE'],
                                                    **{'text-align': 'left'})
                                     .hide_index())
    html = model_performance_df_styled.render()
    imgkit.from_string(html, 'plots/model_performance.png', {'width': 1})
