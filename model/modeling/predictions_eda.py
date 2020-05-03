# Project: Predictions EDA
# Description: Explore trends in player projection model predictions
# Data Sources: Basketball-Reference and ESPN
# Last Updated: 8/7/2019

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting Style
plt.style.use('fivethirtyeight')

if __name__=='__main__':
    # Read in projections
    predictions = pd.read_csv('predictions/predictions.csv')
    # Filter to 2018-2019 and remove three players with outlier predictions due to
    # minimal MP
    predictions = predictions[(predictions['SEASON']=='2018-2019') &
                              (~predictions['BBREF_ID'].isin(['davisty01', 'qizh01', 'siberjo01']))]
    # Transform predictions to long form instead of wide
    long_predictions = pd.melt(predictions[[
                                        'BBREF_ID',
                                        'PLAYER',
                                        'ADVANCED_POSITION_CLUSTER',
                                        'PLUS1_PREDICTION',
                                        'PLUS2_PREDICTION',
                                        'PLUS3_PREDICTION',
                                        'PLUS4_PREDICTION',
                                        'PLUS5_PREDICTION']],
                                        id_vars=['BBREF_ID',
                                                 'PLAYER',
                                                 'ADVANCED_POSITION_CLUSTER'],
                                        value_vars=['PLUS1_PREDICTION',
                                                    'PLUS2_PREDICTION',
                                                    'PLUS3_PREDICTION',
                                                    'PLUS4_PREDICTION',
                                                    'PLUS5_PREDICTION'])
    replace_dict = {'PLUS1_PREDICTION': '2019-2020',
                    'PLUS2_PREDICTION': '2020-2021',
                    'PLUS3_PREDICTION': '2021-2022',
                    'PLUS4_PREDICTION': '2022-2023',
                    'PLUS5_PREDICTION': '2023-2024'}
    long_predictions['SEASON'] = long_predictions['variable'].replace(replace_dict)
    long_predictions.drop('variable', axis=1, inplace=True)
    long_predictions.rename(columns={'value': 'BLEND'}, inplace=True)
    long_predictions = long_predictions[['BBREF_ID',
                                         'PLAYER',
                                         'SEASON',
                                         'ADVANCED_POSITION_CLUSTER',
                                         'BLEND']]

    # Plot predictions by position
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(18, 5), sharex=True, sharey=True)
    sns.violinplot(x='ADVANCED_POSITION_CLUSTER', y='BLEND', data=long_predictions[long_predictions['SEASON']=='2019-2020'], cut=0, ax=axs[0])
    sns.violinplot(x='ADVANCED_POSITION_CLUSTER', y='BLEND', data=long_predictions[long_predictions['SEASON']=='2020-2021'], cut=0, ax=axs[1])
    sns.violinplot(x='ADVANCED_POSITION_CLUSTER', y='BLEND', data=long_predictions[long_predictions['SEASON']=='2021-2022'], cut=0, ax=axs[2])
    sns.violinplot(x='ADVANCED_POSITION_CLUSTER', y='BLEND', data=long_predictions[long_predictions['SEASON']=='2022-2023'], cut=0, ax=axs[3])
    sns.violinplot(x='ADVANCED_POSITION_CLUSTER', y='BLEND', data=long_predictions[long_predictions['SEASON']=='2023-2024'], cut=0, ax=axs[4])
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('Position Cluster')
    axs[3].set_xlabel('')
    axs[4].set_xlabel('')
    axs[0].set_ylabel('RPM/BPM Blend')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[3].set_ylabel('')
    axs[4].set_ylabel('')
    plt.suptitle('Predictions by Position Cluster', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


    # Plot predictions by season
    fig, axs = plt.subplots(figsize=(18, 5))
    sns.swarmplot(x="SEASON", y="BLEND", hue="ADVANCED_POSITION_CLUSTER",
              palette=['tab:blue', sns.xkcd_rgb["pale red"], '#e5ae38'], data=long_predictions)
    plt.suptitle('Predictions by Season', fontsize=20)
    plt.legend(prop={'size': 8}, title='Position Cluster')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
